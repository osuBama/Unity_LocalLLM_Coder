"""Helpers for setup.ps1 (and Linux users): testable logic lives here.

    python -m app.setup_tools detect-gpus
    python -m app.setup_tools plan --primary-gb 12 --memory-gb 8
    python -m app.setup_tools set-config config/config.yaml ollama.primary.model=qwen3:14b ...
    python -m app.setup_tools fit-context --base-url http://127.0.0.1:11434 --model qwen3:14b --start 16384
    python -m app.setup_tools openclaw-patch --model qwen3:14b --ctx 16384

Every command prints one JSON object on stdout (human messages go to stderr).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ GPUs
def parse_nvidia_smi(csv_text: str) -> list[dict]:
    """Parse `nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used --format=csv,noheader,nounits`."""
    gpus = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        total = int(float(parts[3]))
        used = int(float(parts[4])) if len(parts) > 4 and parts[4].replace(".", "").isdigit() else 0
        gpus.append({"index": int(parts[0]), "name": parts[1], "uuid": parts[2],
                     "vram_mib": total, "used_mib": used, "vram_gb": round(total / 1024, 1)})
    return gpus


def detect_gpus() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"ok": False, "error": "nvidia-smi not found (NVIDIA driver missing, or not an NVIDIA system)",
                "gpus": []}
    try:
        out = subprocess.run([exe, "--query-gpu=index,name,uuid,memory.total,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": str(e), "gpus": []}
    gpus = parse_nvidia_smi(out.stdout)
    return {"ok": bool(gpus), "gpus": gpus, "error": None if gpus else out.stderr.strip()[:300]}


def assign_roles(gpus: list[dict], primary_uuid: str | None = None, memory_uuid: str | None = None) -> dict:
    """Larger VRAM = primary, unless UUIDs are given. One GPU = single mode."""
    by_uuid = {g["uuid"]: g for g in gpus}
    if primary_uuid and primary_uuid not in by_uuid:
        raise ValueError(f"primary GPU {primary_uuid} not found")
    if memory_uuid and memory_uuid not in by_uuid:
        raise ValueError(f"memory GPU {memory_uuid} not found")
    if not gpus:
        raise ValueError("no GPUs detected")
    if len(gpus) == 1:
        return {"mode": "single", "primary": gpus[0], "memory": gpus[0]}
    ordered = sorted(gpus, key=lambda g: g["vram_mib"], reverse=True)
    primary = by_uuid.get(primary_uuid) if primary_uuid else None
    memory = by_uuid.get(memory_uuid) if memory_uuid else None
    primary = primary or next(g for g in ordered if g is not memory)
    memory = memory or next(g for g in ordered if g is not primary)
    if primary["uuid"] == memory["uuid"]:
        raise ValueError("primary and memory GPU are the same card")
    return {"mode": "dual", "primary": primary, "memory": memory}


# ------------------------------------------------------------------ plan
# (min VRAM GB, model, num_ctx) - first row whose VRAM fits wins. Q4_K_M sizes, q8_0 KV cache.
PRIMARY_TABLE = [(28, "qwen3:32b", 32768), (22, "qwen3:32b", 16384), (15, "qwen3:14b", 32768),
                 (11, "qwen3:14b", 16384), (7.5, "qwen3:8b", 8192), (5.5, "qwen3:4b", 8192),
                 (0, "qwen3:1.7b", 8192)]
MEMORY_TABLE = [(11, "qwen3:8b", 16384), (7.5, "qwen3:8b", 8192), (5.5, "qwen3:4b", 8192),
                (0, "qwen3:1.7b", 8192)]
SINGLE_TABLE = [(22, "qwen3:14b", 32768, "qwen3:4b", 8192), (15, "qwen3:14b", 16384, "qwen3:4b", 8192),
                (11, "qwen3:8b", 16384, "qwen3:1.7b", 8192), (7.5, "qwen3:4b", 8192, "qwen3:1.7b", 4096),
                (0, "qwen3:1.7b", 8192, "qwen3:1.7b", 4096)]


def plan(primary_gb: float, memory_gb: float | None = None, *, desktop_gb: float = 0.0) -> dict:
    """Model/context suggestions. desktop_gb = VRAM already used on the primary (monitors, apps)."""
    p_avail = max(0.0, primary_gb - desktop_gb)
    if memory_gb is None:
        _, pm, pc, mm, mc = next(r for r in SINGLE_TABLE if p_avail >= r[0])
        mode = "single"
    else:
        _, pm, pc = next(r for r in PRIMARY_TABLE if p_avail >= r[0])
        _, mm, mc = next(r for r in MEMORY_TABLE if memory_gb >= r[0])
        mode = "dual"
    budget = budgets(pc)
    return {"mode": mode, "primary_model": pm, "primary_ctx": pc, "memory_model": mm, "memory_ctx": mc,
            **budget}


def budgets(primary_ctx: int) -> dict:
    """Memory budget ~15% of the primary window; the cached base gets ~40% of that."""
    mem = int(min(6000, max(1000, round(primary_ctx * 0.15 / 100) * 100)))
    base = int(round(mem * 0.4 / 100) * 100)
    return {"memory_budget": mem, "memory_base_budget": base}


# ------------------------------------------------------------- config edit
_KEY_RE = re.compile(r"^(\s*)([A-Za-z_][\w-]*):(.*)$")


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ", ".join(_yaml_scalar(v) for v in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)   # double-quoted, backslashes escaped


def _split_value_comment(rest: str) -> tuple[str, str]:
    """' "a # b"   # note' -> ('"a # b"', '   # note'), respecting quotes."""
    q = None
    for i, ch in enumerate(rest):
        if q:
            if ch == "\\" and q == '"':
                continue
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch == "#" and (i == 0 or rest[i - 1].isspace()):
            j = i
            while j > 0 and rest[j - 1].isspace():
                j -= 1
            return rest[:j], rest[j:]
    return rest.rstrip(), ""


def set_yaml_values(text: str, updates: dict[str, Any]) -> str:
    """Set dotted keys in a simple block-style YAML file, keeping comments and layout.

    Existing keys are edited in place (trailing comments kept). Missing keys are
    inserted at the end of their parent section, or a new section is appended.
    """
    lines = text.splitlines()
    for dotted, value in updates.items():
        target = dotted.split(".")
        stack: list[tuple[int, str]] = []
        found = False
        parent_end: dict[tuple[str, ...], int] = {}
        parent_indent: dict[tuple[str, ...], int] = {}
        for i, line in enumerate(lines):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = _KEY_RE.match(line)
            if not m:
                continue
            indent = len(m.group(1))
            while stack and stack[-1][0] >= indent:
                stack.pop()
            path = tuple(k for _, k in stack) + (m.group(2),)
            for depth in range(len(path)):
                parent_end[path[:depth]] = i
            parent_indent.setdefault(path[:-1], indent)
            if list(path) == target:
                value_part, comment = _split_value_comment(m.group(3))
                if value_part.strip() == "" and i + 1 < len(lines):
                    raise ValueError(f"{dotted} is a section, not a value")
                lines[i] = f"{m.group(1)}{m.group(2)}: {_yaml_scalar(value)}{comment}"
                found = True
                break
            stack.append((indent, m.group(2)))
        if found:
            continue
        # Insert under the deepest existing parent.
        for depth in range(len(target) - 1, -1, -1):
            parent = tuple(target[:depth])
            if depth == 0 or parent in parent_end:
                break
        if depth == 0:
            insert_at, base_indent = len(lines), 0
            if lines and lines[-1].strip():
                lines.append("")
                insert_at += 1
        else:
            insert_at = parent_end[parent] + 1
            base_indent = parent_indent.get(parent, depth * 2)
        new = []
        for k, key in enumerate(target[depth:]):
            pad = " " * (base_indent + 2 * k)
            last = k == len(target[depth:]) - 1
            new.append(f"{pad}{key}: {_yaml_scalar(value)}" if last else f"{pad}{key}:")
        lines[insert_at:insert_at] = new
    return "\n".join(lines) + "\n"


def set_config(path: Path, updates: dict[str, Any]) -> dict:
    """Edit config.yaml in place with a timestamped backup; roll back if it no longer validates."""
    from .config import load_config
    path = Path(path)
    original = path.read_text(encoding="utf-8")
    backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    backup.write_text(original, encoding="utf-8")
    path.write_text(set_yaml_values(original, updates), encoding="utf-8")
    try:
        load_config(path)
    except Exception as e:
        path.write_text(original, encoding="utf-8")
        return {"ok": False, "error": f"new config does not validate, restored: {e}", "backup": str(backup)}
    return {"ok": True, "backup": str(backup), "updated": sorted(updates)}


def parse_assignment(item: str) -> tuple[str, Any]:
    key, _, raw = item.partition("=")
    if not key or not _:
        raise ValueError(f"expected key=value, got {item!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.strip(), value


# ---------------------------------------------------------- context fitting
def _ps_entry(client: httpx.Client, model: str) -> dict | None:
    for m in client.get("/api/ps").json().get("models", []) or []:
        if m.get("name") == model or m.get("model") == model:
            return m
    return None


def fit_context(base_url: str, model: str, start: int, *, minimum: int = 4096, keep: list[str] | None = None,
                embed_model: str | None = None, transport: httpx.BaseTransport | None = None,
                timeout: float = 600) -> dict:
    """Largest num_ctx (from `start` downwards) at which `model` loads 100% on GPU.

    `keep`: models that must stay loaded alongside (single-GPU mode); they are
    re-checked after each attempt.
    """
    candidates = []
    c = start
    while c >= minimum:
        candidates.append(c)
        c = int(c * 0.75) // 1024 * 1024
    if not candidates or candidates[-1] != minimum:
        candidates.append(minimum)
    tried = []
    keep = list(keep or [])
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport) as client:
        if embed_model:
            # The embedding model shares this GPU: load it first so the fit leaves room for it.
            try:
                client.post("/api/embed", json={"model": embed_model, "input": "warm-up",
                                                "keep_alive": "10m"}).raise_for_status()
                keep.append(embed_model)
            except httpx.HTTPError as e:
                return {"ok": False, "model": model, "num_ctx": None, "tried": [],
                        "error": f"could not load embedding model {embed_model}: {e}"}
        for ctx in candidates:
            try:
                r = client.post("/api/generate", json={"model": model, "prompt": "", "keep_alive": "10m",
                                                       "options": {"num_ctx": ctx}})
                r.raise_for_status()
                entry = _ps_entry(client, model)
            except httpx.HTTPError as e:
                tried.append({"num_ctx": ctx, "error": str(e)[:200]})
                continue
            if entry is None:
                tried.append({"num_ctx": ctx, "error": "model not listed in /api/ps"})
                continue
            size, vram = entry.get("size") or 0, entry.get("size_vram") or 0
            others_ok = all((_ps_entry(client, k) or {}).get("size_vram", 0) >= (_ps_entry(client, k) or {}).get("size", 1)
                            for k in (keep or []))
            pct = round(100 * vram / size, 1) if size else 0
            tried.append({"num_ctx": ctx, "gpu_percent": pct, "others_on_gpu": others_ok})
            if size and vram >= size and others_ok:
                return {"ok": True, "model": model, "num_ctx": ctx, "vram_gb": round(vram / 2**30, 2),
                        "tried": tried}
            _err(f"  {model} at num_ctx={ctx}: {pct}% on GPU, trying smaller")
            client.post("/api/generate", json={"model": model, "keep_alive": 0})   # unload before retry
    return {"ok": False, "model": model, "num_ctx": None, "tried": tried,
            "error": f"{model} does not fit fully on the GPU even at num_ctx={minimum}; choose a smaller model"}


# --------------------------------------------------------------- OpenClaw
def openclaw_patch(model: str, ctx: int, base_url: str = "http://127.0.0.1:8000",
                   reasoning: bool = True, max_tokens: int = 4096) -> dict:
    """The OpenClaw config patch (JSON) that points its Ollama provider at the orchestrator."""
    return {"models": {"providers": {"ollama": {
        "baseUrl": base_url, "api": "ollama", "apiKey": "ollama-local", "timeoutSeconds": 600,
        "models": [{"id": model, "name": model, "reasoning": reasoning, "input": ["text"],
                    "contextWindow": ctx, "maxTokens": max_tokens, "params": {"keep_alive": "30m"}}]}}},
        "agents": {"defaults": {"model": {"primary": f"ollama/{model}"}}},
        # Reference-library tools (docs_search / docs_lookup) served by the orchestrator.
        "mcp": {"servers": {"local-docs": {"url": base_url.rstrip("/") + "/mcp", "transport": "streamable-http"}}}}


# -------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.setup_tools")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("detect-gpus")
    s.add_argument("--primary-uuid")
    s.add_argument("--memory-uuid")
    s = sp.add_parser("plan")
    s.add_argument("--primary-gb", type=float, required=True)
    s.add_argument("--memory-gb", type=float)
    s.add_argument("--desktop-gb", type=float, default=0.0)
    s = sp.add_parser("budgets")
    s.add_argument("--ctx", type=int, required=True)
    s = sp.add_parser("set-config")
    s.add_argument("path")
    s.add_argument("assignments", nargs="*", help="key=value (value parsed as JSON if possible)")
    s.add_argument("--json-file", help="JSON object of dotted key -> value (safe from shell quoting)")
    s = sp.add_parser("fit-context")
    s.add_argument("--base-url", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--start", type=int, required=True)
    s.add_argument("--minimum", type=int, default=4096)
    s.add_argument("--keep", action="append", default=[])
    s.add_argument("--embed-model", help="embedding model sharing this GPU (loaded first)")
    s = sp.add_parser("openclaw-patch")
    s.add_argument("--model", required=True)
    s.add_argument("--ctx", type=int, required=True)
    s.add_argument("--base-url", default="http://127.0.0.1:8000")
    s.add_argument("--provider-only", action="store_true", help="print only models.providers.ollama")
    a = p.parse_args(argv)

    try:
        if a.cmd == "detect-gpus":
            out = detect_gpus()
            if out["ok"]:
                out["roles"] = assign_roles(out["gpus"], a.primary_uuid, a.memory_uuid)
        elif a.cmd == "plan":
            out = plan(a.primary_gb, a.memory_gb, desktop_gb=a.desktop_gb)
        elif a.cmd == "budgets":
            out = budgets(a.ctx)
        elif a.cmd == "set-config":
            updates = dict(parse_assignment(x) for x in a.assignments)
            if a.json_file:
                updates.update(json.loads(Path(a.json_file).read_text(encoding="utf-8-sig")))
            if not updates:
                raise ValueError("nothing to set")
            out = set_config(Path(a.path), updates)
        elif a.cmd == "fit-context":
            out = fit_context(a.base_url, a.model, a.start, minimum=a.minimum, keep=a.keep,
                              embed_model=a.embed_model)
        else:
            out = openclaw_patch(a.model, a.ctx, a.base_url)
            if a.provider_only:
                out = out["models"]["providers"]["ollama"]
    except Exception as e:
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
