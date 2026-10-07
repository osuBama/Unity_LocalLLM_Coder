"""MCP server (Streamable HTTP, JSON responses) exposing the reference libraries as tools.

OpenClaw (or any MCP client) points at http://127.0.0.1:8000/mcp. Tools:
  docs_search(query, library?, max_results?)   hybrid search over enabled libraries
  docs_lookup(symbol, library?)                exact symbol: "Rigidbody.AddForce", "AddForce"
  docs_libraries()                             what is indexed (name, version, description)

Read-only. Requests carrying a browser Origin that is not local are refused
(protection against DNS rebinding, as the MCP spec recommends for local servers).
"""
from __future__ import annotations

import json
import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from . import __version__
from .references import format_hits

PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]

TOOLS = [
    {"name": "docs_search",
     "description": "Search the configured documentation and code libraries (e.g. Unity scripting reference, "
                    "packages). Use it before relying on an API you are not sure about, or when the version matters.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "What you need, e.g. 'move kinematic rigidbody with interpolation'"},
         "library": {"type": "string", "description": "Optional library name to restrict the search"},
         "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 4}},
         "required": ["query"]}},
    {"name": "docs_lookup",
     "description": "Look up an exact class, method or property, e.g. 'Rigidbody.AddForce' or 'NavMeshAgent'. "
                    "Returns its documentation and signature.",
     "inputSchema": {"type": "object", "properties": {
         "symbol": {"type": "string"}, "library": {"type": "string"}}, "required": ["symbol"]}},
    {"name": "docs_libraries",
     "description": "List the documentation libraries that are available, with versions.",
     "inputSchema": {"type": "object", "properties": {}}},
]


def _local_origin(origin: str | None) -> bool:
    if not origin:
        return True                      # non-browser clients send no Origin
    host = urlparse(origin).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")


def build_mcp_router(orch) -> APIRouter:
    router = APIRouter()
    refs = orch.references

    def result(id_, payload):
        return {"jsonrpc": "2.0", "id": id_, "result": payload}

    def error(id_, code, message):
        return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}

    async def call_tool(name: str, args: dict) -> dict:
        cap = orch.config.references.tool_max_tokens
        libs = [args["library"]] if isinstance(args.get("library"), str) and args.get("library") else None
        if name == "docs_libraries":
            items = [l for l in refs.libraries() if l["enabled"]]
            text = "\n".join(f"- {l['name']}{(' ' + l['version']) if l['version'] else ''}: "
                             f"{l['description'] or l['source_path']} ({l['chunks']} sections)" for l in items) \
                or "No documentation libraries are enabled."
            return {"content": [{"type": "text", "text": text}]}
        if name == "docs_search":
            q = str(args.get("query") or "").strip()
            if not q:
                return {"content": [{"type": "text", "text": "query is required"}], "isError": True}
            n = max(1, min(10, int(args.get("max_results") or 4)))
            qvec = await orch.query_vector(q)
            hits = refs.search(q, qvec, libraries=libs, limit=n)
            return {"content": [{"type": "text", "text": format_hits(hits, cap)}]}
        if name == "docs_lookup":
            sym = str(args.get("symbol") or "").strip()
            hits = refs.lookup(sym, libraries=libs, limit=4)
            if not hits:   # fall back to search so a near-miss still helps
                hits = refs.search(sym, await orch.query_vector(sym), libraries=libs, limit=3)
            return {"content": [{"type": "text", "text": format_hits(hits, cap)}]}
        return {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}

    @router.post("/mcp")
    async def mcp(request: Request):
        if not _local_origin(request.headers.get("origin")):
            return JSONResponse({"error": "origin not allowed"}, status_code=403)
        if refs is None or not orch.config.references.mcp_enabled:
            return JSONResponse(error(None, -32601, "reference tools are disabled"), status_code=404)
        try:
            msg = json.loads(await request.body() or b"{}")
        except json.JSONDecodeError:
            return JSONResponse(error(None, -32700, "parse error"), status_code=400)
        if isinstance(msg, list):
            return JSONResponse(error(None, -32600, "batches are not supported"), status_code=400)
        method, id_ = msg.get("method"), msg.get("id")
        if id_ is None:                                   # notification (e.g. notifications/initialized)
            return Response(status_code=202)
        if method == "initialize":
            asked = (msg.get("params") or {}).get("protocolVersion")
            version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            body = result(id_, {"protocolVersion": version,
                                "capabilities": {"tools": {"listChanged": False}},
                                "serverInfo": {"name": "local-ai-docs", "version": __version__},
                                "instructions": "Documentation search for the user's configured libraries."})
            return JSONResponse(body, headers={"Mcp-Session-Id": uuid.uuid4().hex})
        if method == "ping":
            return JSONResponse(result(id_, {}))
        if method == "tools/list":
            return JSONResponse(result(id_, {"tools": TOOLS}))
        if method == "tools/call":
            p = msg.get("params") or {}
            try:
                out = await call_tool(str(p.get("name")), p.get("arguments") or {})
            except Exception as e:
                out = {"content": [{"type": "text", "text": f"documentation search failed: {e}"}], "isError": True}
            return JSONResponse(result(id_, out))
        return JSONResponse(error(id_, -32601, f"method not found: {method}"))

    @router.get("/mcp")
    async def mcp_get():
        return Response(status_code=405, headers={"Allow": "POST"})

    @router.delete("/mcp")
    async def mcp_delete():
        return Response(status_code=200)

    return router
