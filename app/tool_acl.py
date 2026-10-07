"""Tool access control: decide which tool definitions the model sees.

Every tool the client offers (OpenClaw's built-ins and every MCP server it connects to)
passes through the proxy in the request's `tools` field. This module:

  * registers them automatically (name, description, schema size, calls, last seen),
  * groups them by *source* (pattern rules first, then the name prefix, e.g.
    "unity__read_console" -> "unity"),
  * applies a mode per tool (or inherited from its source, or the default):
      off        removed from the request; calls to it are blocked
      automatic  full definition on every request (as without this feature)
      on_demand  listed in one compact `load_tools` catalog; the model loads what it
                 needs, the proxy answers `load_tools` itself and re-asks the model with
                 the full definitions. Loaded tools stay loaded for the conversation.

The model never sees this module's state, only the resulting tool list.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import threading
import time
from collections import OrderedDict

from .util import estimate_tokens, now_iso

MODES = ("off", "automatic", "on_demand")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_registry (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    schema TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    schema_tokens INTEGER NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    calls INTEGER NOT NULL DEFAULT 0,
    blocked INTEGER NOT NULL DEFAULT 0,
    loads INTEGER NOT NULL DEFAULT 0,
    mode TEXT,                    -- NULL = inherit from source
    source_override TEXT
);
CREATE TABLE IF NOT EXISTS tool_sources (
    source TEXT PRIMARY KEY,
    mode TEXT                     -- NULL = default mode
);
CREATE TABLE IF NOT EXISTS tool_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT NOT NULL,        -- fnmatch glob on the tool name, e.g. unity_*
    source TEXT NOT NULL
);
"""


def _name(tool: dict) -> str:
    return str(((tool or {}).get("function") or {}).get("name") or "")


def _desc(tool: dict) -> str:
    return str(((tool or {}).get("function") or {}).get("description") or "")


def heuristic_source(name: str) -> str:
    """Best guess before any rules: MCP-style prefixes, our own docs tools, else the client."""
    n = name
    if n.startswith("mcp__"):
        n = n[5:]
    if "__" in n:
        return n.rsplit("__", 1)[0].replace("__", "/")
    if name in ("docs_search", "docs_lookup", "docs_libraries"):
        return "local-docs"
    return "client"


def one_line(text: str, limit: int = 110) -> str:
    t = " ".join((text or "").split())
    cut = t.split(". ")[0]
    return (cut if len(cut) <= limit else cut[:limit - 1] + "…") or "(no description)"


class ToolACL:
    def __init__(self, orch):
        self.orch = orch
        self.db = orch.db
        self.cfg = orch.config.tools
        with self.db.connect() as c:
            c.executescript(SCHEMA)
        self._seen: dict[str, tuple[str, float]] = {}     # name -> (schema hash, last write time)
        self._gen = None
        self._state: dict | None = None
        self._loaded: OrderedDict[str, set[str]] = OrderedDict()   # conversation -> loaded on-demand tools
        self._lock = threading.Lock()

    # ------------------------------------------------------------- state
    def _bump(self) -> None:
        self.db.kv_set("tools.generation", str(time.time_ns()))

    def _load_state(self) -> dict:
        gen = self.db.kv_get("tools.generation")
        with self._lock:
            if self._state is None or gen != self._gen:
                with self.db.connect() as c:
                    tools = {r["name"]: dict(r) for r in c.execute(
                        "SELECT name, mode, source_override FROM tool_registry")}
                    sources = {r["source"]: r["mode"] for r in c.execute("SELECT * FROM tool_sources")}
                    rules = [dict(r) for r in c.execute("SELECT * FROM tool_rules ORDER BY id")]
                self._state = {"tools": tools, "sources": sources, "rules": rules,
                               "default": self.db.kv_get("tools.default_mode") or self.cfg.default_mode}
                self._gen = gen
            return self._state

    def default_mode(self) -> str:
        return self._load_state()["default"]

    def source_of(self, name: str) -> str:
        st = self._load_state()
        t = st["tools"].get(name) or {}
        if t.get("source_override"):
            return t["source_override"]
        for r in st["rules"]:
            if fnmatch.fnmatchcase(name, r["pattern"]):
                return r["source"]
        return heuristic_source(name)

    def mode_of(self, name: str) -> tuple[str, str]:
        """(mode, where it came from: tool / source / default)."""
        st = self._load_state()
        t = st["tools"].get(name) or {}
        if t.get("mode") in MODES:
            return t["mode"], "tool"
        src_mode = st["sources"].get(self.source_of(name))
        if src_mode in MODES:
            return src_mode, "source"
        return st["default"], "default"

    # ----------------------------------------------------------- observe
    def observe(self, tools: list[dict]) -> None:
        """Register tools seen in a request (writes only when new, changed, or stale by 5 minutes)."""
        now = time.time()
        changed = False
        rows = []
        for t in tools or []:
            n = _name(t)
            if not n or n == self.cfg.meta_tool_name:
                continue
            raw = json.dumps(t, ensure_ascii=False, sort_keys=True)
            h = hashlib.sha1(raw.encode()).hexdigest()[:16]
            prev = self._seen.get(n)
            if prev and prev[0] == h and now - prev[1] < 300:
                continue
            self._seen[n] = (h, now)
            rows.append((n, _desc(t)[:2000], raw, h, estimate_tokens(raw), now_iso()))
        if not rows:
            return
        with self.db.connect() as c:
            for n, d, raw, h, tok, ts in rows:
                cur = c.execute("UPDATE tool_registry SET description=?, schema=?, schema_hash=?, schema_tokens=?, "
                                "last_seen=? WHERE name=?", (d, raw, h, tok, ts, n))
                if cur.rowcount == 0:
                    c.execute("INSERT INTO tool_registry (name, description, schema, schema_hash, schema_tokens, "
                              "first_seen, last_seen) VALUES (?,?,?,?,?,?,?)", (n, d, raw, h, tok, ts, ts))
                    changed = True
        if changed:
            self._bump()

    # ---------------------------------------------------------- per request
    def loaded_for(self, conversation_id: str) -> set[str]:
        return self._loaded.get(conversation_id, set())

    def load(self, conversation_id: str, names: list[str], offered: list[dict]) -> tuple[list[str], list[str]]:
        """Mark on-demand tools as loaded for this conversation. Returns (loaded, refused)."""
        available = {_name(t) for t in offered if self.mode_of(_name(t))[0] == "on_demand"}
        ok = [n for n in names if n in available]
        bad = [n for n in names if n not in available]
        if ok:
            s = self._loaded.setdefault(conversation_id, set())
            s.update(ok)
            self._loaded.move_to_end(conversation_id)
            while len(self._loaded) > 512:
                self._loaded.popitem(last=False)
            with self.db.connect() as c:
                c.executemany("UPDATE tool_registry SET loads = loads + 1 WHERE name=?", [(n,) for n in ok])
        return ok, bad

    def effective(self, conversation_id: str, tools: list[dict]) -> tuple[list[dict], dict]:
        """The tool list the model gets, plus a summary for metrics."""
        if not self.cfg.enabled or not tools:
            return tools, {}
        loaded = self.loaded_for(conversation_id)
        kept, catalog, hidden = [], [], []
        for t in tools:
            n = _name(t)
            if not n:
                kept.append(t)
                continue
            mode, _ = self.mode_of(n)
            if mode == "off":
                hidden.append(n)
            elif mode == "on_demand" and n not in loaded:
                catalog.append(t)
            else:
                kept.append(t)
        if catalog:
            kept.append(self.meta_tool(catalog))
        info = {"tools_offered_by_client": len(tools), "tools_sent": len(kept),
                "tools_hidden": len(hidden), "tools_in_catalog": len(catalog),
                "tools_loaded": len(loaded)}
        return kept, info

    def meta_tool(self, catalog: list[dict]) -> dict:
        lines = "\n".join(f"- {_name(t)}: {one_line(_desc(t))}" for t in catalog)
        return {"type": "function", "function": {
            "name": self.cfg.meta_tool_name,
            "description": ("Load more tools before using them. Call this with the names of the tools you need; "
                            "afterwards their full definitions are available and you can call them.\n"
                            f"Available tools:\n{lines}"),
            "parameters": {"type": "object", "properties": {
                "names": {"type": "array", "items": {"type": "string", "enum": [_name(t) for t in catalog]},
                          "description": "Tool names to load"}}, "required": ["names"]}}}

    def classify_calls(self, tool_calls: list) -> tuple[list, list[str], list[str]]:
        """(calls to forward, blocked tool names, names requested via load_tools)."""
        keep, blocked, load = [], [], []
        for tc in tool_calls or []:
            fn = (tc or {}).get("function") or {}
            n = str(fn.get("name") or "")
            if n == self.cfg.meta_tool_name:
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                names = args.get("names") if isinstance(args, dict) else None
                load.extend(str(x) for x in (names if isinstance(names, list) else [names] if names else []))
                continue
            if self.cfg.enabled and self.cfg.block_disabled_calls and n and self.mode_of(n)[0] == "off":
                blocked.append(n)
                continue
            keep.append(tc)
        return keep, blocked, load

    def record(self, called: list[str], blocked: list[str]) -> None:
        if not (called or blocked):
            return
        with self.db.connect() as c:
            c.executemany("UPDATE tool_registry SET calls = calls + 1 WHERE name=?", [(n,) for n in called])
            c.executemany("UPDATE tool_registry SET blocked = blocked + 1 WHERE name=?", [(n,) for n in blocked])

    def blocked_note(self, names: list[str]) -> str:
        return (f"[The orchestrator blocked a call to {', '.join(sorted(set(names)))}: that tool is disabled "
                "in its tool settings. Continue without it or ask the user to enable it.]")

    # ------------------------------------------------------------ admin
    def listing(self) -> dict:
        with self.db.connect() as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM tool_registry ORDER BY name")]
            rules = [dict(r) for r in c.execute("SELECT * FROM tool_rules ORDER BY id")]
            src_modes = {r["source"]: r["mode"] for r in c.execute("SELECT * FROM tool_sources")}
        groups: dict[str, dict] = {}
        for r in rows:
            src = self.source_of(r["name"])
            mode, origin = self.mode_of(r["name"])
            g = groups.setdefault(src, {"source": src, "mode": src_modes.get(src), "tools": [],
                                        "tokens_if_automatic": 0})
            g["tools"].append({"name": r["name"], "description": one_line(r["description"], 160),
                               "schema_tokens": r["schema_tokens"], "calls": r["calls"], "blocked": r["blocked"],
                               "loads": r["loads"], "first_seen": r["first_seen"], "last_seen": r["last_seen"],
                               "mode": r["mode"], "effective_mode": mode, "mode_from": origin,
                               "source_override": r["source_override"]})
            g["tokens_if_automatic"] += r["schema_tokens"]
        sent = sum(t["schema_tokens"] for g in groups.values() for t in g["tools"] if t["effective_mode"] == "automatic")
        cat = sum(estimate_tokens(one_line(t["description"])) + 6 for g in groups.values() for t in g["tools"]
                  if t["effective_mode"] == "on_demand")
        return {"enabled": self.cfg.enabled, "default_mode": self.default_mode(), "rules": rules,
                "sources": sorted(groups.values(), key=lambda g: g["source"]),
                "tokens_per_request": {"automatic": sent, "catalog": cat + (60 if cat else 0),
                                       "all_if_automatic": sum(g["tokens_if_automatic"] for g in groups.values())}}

    def set_mode(self, *, tool: str | None = None, source: str | None = None, mode: str | None) -> None:
        if mode is not None and mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)} (or empty to inherit)")
        with self.db.connect() as c:
            if tool:
                if c.execute("UPDATE tool_registry SET mode=? WHERE name=?", (mode, tool)).rowcount == 0:
                    raise KeyError(f"unknown tool {tool!r}")
            elif source:
                c.execute("INSERT INTO tool_sources (source, mode) VALUES (?,?) "
                          "ON CONFLICT(source) DO UPDATE SET mode=excluded.mode", (source, mode))
            else:
                raise ValueError("give a tool or a source")
        self._bump()

    def set_default(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        self.db.kv_set("tools.default_mode", mode)
        self._bump()

    def set_source(self, tool: str, source: str | None) -> None:
        with self.db.connect() as c:
            if c.execute("UPDATE tool_registry SET source_override=? WHERE name=?",
                         (source or None, tool)).rowcount == 0:
                raise KeyError(f"unknown tool {tool!r}")
        self._bump()

    def add_rule(self, pattern: str, source: str) -> int:
        if not pattern.strip() or not source.strip():
            raise ValueError("pattern and source are required")
        with self.db.connect() as c:
            rid = int(c.execute("INSERT INTO tool_rules (pattern, source) VALUES (?,?)",
                                (pattern.strip(), source.strip())).lastrowid)
        self._bump()
        return rid

    def remove_rule(self, rule_id: int) -> bool:
        with self.db.connect() as c:
            ok = c.execute("DELETE FROM tool_rules WHERE id=?", (rule_id,)).rowcount > 0
        self._bump()
        return ok

    def forget(self, tool: str | None = None, older_than_days: int | None = None) -> int:
        with self.db.connect() as c:
            if tool:
                n = c.execute("DELETE FROM tool_registry WHERE name=?", (tool,)).rowcount
            else:
                from datetime import datetime, timedelta
                cutoff = (datetime.now().astimezone() - timedelta(days=older_than_days or 30)).isoformat(timespec="seconds")
                n = c.execute("DELETE FROM tool_registry WHERE last_seen < ?", (cutoff,)).rowcount
        self._seen.clear()
        self._bump()
        return n
