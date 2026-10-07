import json

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.tool_acl import heuristic_source

H = {"X-AI-Client": "1"}


def tool(name, desc="Does a thing.", props=None):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props or {"x": {"type": "string"}}}}}


TOOLS = [tool("read_file", "Read a file from the workspace."), tool("shell", "Run a shell command."),
         tool("unity__read_console", "Read the Unity Editor console. Returns errors and warnings."),
         tool("unity__execute_code", "Execute arbitrary C# in the Editor."),
         tool("unity__manage_vfx", "Create and edit VFX Graph systems. " + "Lots of options. " * 80),
         tool("docs_search", "Search documentation.")]


def call(name, args=None):
    return [{"function": {"name": name, "arguments": args or {}}}]


@pytest.fixture
def env(cfg, fakes):
    primary, memory = fakes
    app = create_app(cfg, primary_transport=primary.transport, memory_transport=memory.transport, console_logs=False)
    with TestClient(app) as client:
        yield client, app.state.orch, primary


def chat(client, text="please check the console", stream=False, cid="c1", extra=None):
    body = {"model": "qwen3:14b", "stream": stream, "tools": TOOLS,
            "messages": [{"role": "user", "content": text}] + (extra or [])}
    return client.post("/api/chat", json=body, headers={"X-Conversation-Id": cid})


def sent_tool_names(primary, i=-1):
    return [t["function"]["name"] for t in primary.requests[i].get("tools") or []]


def test_heuristic_sources():
    assert heuristic_source("unity__read_console") == "unity"
    assert heuristic_source("mcp__github__create_issue") == "github"
    assert heuristic_source("docs_lookup") == "local-docs"
    assert heuristic_source("read_file") == "client"


def test_registry_groups_rules_and_overrides(env):
    client, orch, primary = env
    chat(client)
    acl = orch.tools_acl
    groups = {g["source"]: [t["name"] for t in g["tools"]] for g in acl.listing()["sources"]}
    assert groups["unity"] == ["unity__execute_code", "unity__manage_vfx", "unity__read_console"]
    assert groups["client"] == ["read_file", "shell"] and groups["local-docs"] == ["docs_search"]
    acl.add_rule("shell", "system")
    acl.set_source("read_file", "workspace")
    groups = {g["source"]: [t["name"] for t in g["tools"]] for g in acl.listing()["sources"]}
    assert groups["system"] == ["shell"] and groups["workspace"] == ["read_file"] and "client" not in groups


def test_mode_resolution_and_request_filtering(env):
    client, orch, primary = env
    chat(client)
    assert sent_tool_names(primary) == [t["function"]["name"] for t in TOOLS]       # default: automatic
    acl = orch.tools_acl
    acl.set_mode(source="unity", mode="on_demand")
    acl.set_mode(tool="unity__execute_code", mode="off")
    acl.set_mode(tool="shell", mode="off")
    assert acl.mode_of("unity__read_console") == ("on_demand", "source")
    assert acl.mode_of("unity__execute_code") == ("off", "tool")
    assert acl.mode_of("read_file") == ("automatic", "default")
    chat(client, cid="c2")
    names = sent_tool_names(primary)
    assert names == ["read_file", "docs_search", "load_tools"]
    meta = primary.requests[-1]["tools"][-1]["function"]
    assert meta["parameters"]["properties"]["names"]["items"]["enum"] == ["unity__read_console", "unity__manage_vfx"]
    assert "unity__read_console: Read the Unity Editor console" in meta["description"]
    assert "execute_code" not in json.dumps(primary.requests[-1]["tools"])       # off: invisible
    m = client.get("/metrics").json()["recent_requests"][-1]
    assert m["tools_hidden"] == 2 and m["tools_in_catalog"] == 2 and m["tools_sent"] == 3
    acl.set_default("on_demand")
    assert acl.mode_of("read_file") == ("on_demand", "default")


def test_blocked_call_never_reaches_client(env):
    client, orch, primary = env
    orch.tools_acl.observe(TOOLS)
    orch.tools_acl.set_mode(tool="unity__execute_code", mode="off")
    primary.tool_call_script = [call("unity__execute_code", {"code": "System.IO.File.Delete(x)"})]
    r = chat(client).json()
    assert "tool_calls" not in r["message"] and "blocked a call to unity__execute_code" in r["message"]["content"]
    assert client.get("/metrics").json()["recent_requests"][-1]["tools_blocked"] == 1
    primary.tool_call_script = [call("unity__execute_code") + call("read_file", {"path": "a.cs"})]
    r = chat(client, cid="c3").json()
    assert [t["function"]["name"] for t in r["message"]["tool_calls"]] == ["read_file"]     # the allowed one survives


def test_on_demand_load_round_non_stream(env):
    client, orch, primary = env
    orch.tools_acl.observe(TOOLS)
    orch.tools_acl.set_mode(source="unity", mode="on_demand")
    primary.tool_call_script = [call("load_tools", {"names": ["unity__read_console"]}), call("unity__read_console")]
    n0 = len(primary.requests)
    r = chat(client).json()
    assert len(primary.requests) - n0 == 2                              # one internal round
    second = primary.requests[-1]
    assert "unity__read_console" in [t["function"]["name"] for t in second["tools"]]
    assert second["messages"][-1]["role"] == "tool" and "Loaded: unity__read_console" in second["messages"][-1]["content"]
    assert [t["function"]["name"] for t in r["message"]["tool_calls"]] == ["unity__read_console"]   # client sees the real call
    assert client.get("/metrics").json()["recent_requests"][-1]["tool_load_rounds"] == 1
    primary.tool_call_script = []
    chat(client, text="again")                                           # same conversation: stays loaded
    names = sent_tool_names(primary)
    assert "unity__read_console" in names and "load_tools" in names     # vfx still in the catalog
    meta = primary.requests[-1]["tools"][-1]["function"]
    assert meta["parameters"]["properties"]["names"]["items"]["enum"] == ["unity__execute_code", "unity__manage_vfx"]
    chat(client, text="new conversation", cid="other")
    assert "unity__read_console" not in sent_tool_names(primary)       # loaded per conversation only


def test_on_demand_load_round_streaming(env):
    client, orch, primary = env
    orch.tools_acl.observe(TOOLS)
    orch.tools_acl.set_mode(source="unity", mode="on_demand")
    primary.tool_call_script = [call("load_tools", {"names": ["unity__read_console", "nope"]}), None]
    primary.reply = "The console shows two errors"
    with client.stream("POST", "/api/chat", json={"model": "m", "stream": True, "tools": TOOLS,
                                                  "messages": [{"role": "user", "content": "check console"}]}) as r:
        lines = [json.loads(l) for l in r.iter_lines() if l]
    assert not any(l["message"].get("tool_calls") for l in lines)           # load_tools never shown to the client
    assert sum(1 for l in lines if l.get("done")) == 1 and lines[-1]["done"]
    assert "".join(l["message"]["content"] for l in lines).strip() == "The console shows two errors"
    tool_msg = primary.requests[-1]["messages"][-1]["content"]
    assert "Loaded: unity__read_console" in tool_msg and "Not available: nope" in tool_msg


def test_streaming_blocked_and_round_limit(env):
    client, orch, primary = env
    orch.tools_acl.observe(TOOLS)
    orch.tools_acl.set_mode(source="unity", mode="on_demand")
    orch.config.tools.max_load_rounds = 0
    primary.tool_call_script = [call("load_tools", {"names": ["unity__read_console"]})]
    with client.stream("POST", "/api/chat", json={"model": "m", "stream": True, "tools": TOOLS,
                                                  "messages": [{"role": "user", "content": "x"}]}) as r:
        lines = [json.loads(l) for l in r.iter_lines() if l]
    text = "".join(l["message"].get("content", "") for l in lines)
    assert "blocked a call to load_tools" in text and lines[-1]["done"]


def test_size_estimate_uses_effective_tools(env):
    client, orch, primary = env
    orch.config.ollama.primary.num_ctx = 16384
    chat(client, cid="a")
    full = client.get("/metrics").json()["recent_requests"][-1]["est_prompt_tokens"]
    orch.tools_acl.set_mode(source="unity", mode="off")
    chat(client, cid="b")
    assert client.get("/metrics").json()["recent_requests"][-1]["est_prompt_tokens"] < full - 300


def test_console_and_cli(env, tmp_path, capsys, cfg):
    client, orch, primary = env
    chat(client)
    lst = client.get("/ui/api/tools").json()
    assert {g["source"] for g in lst["sources"]} == {"client", "unity", "local-docs"}
    assert lst["tokens_per_request"]["all_if_automatic"] > 300
    assert client.post("/ui/api/tools/mode", json={"source": "unity", "mode": "on_demand"}, headers=H).status_code == 200
    assert client.post("/ui/api/tools/mode", json={"tool": "shell", "mode": "off"}, headers=H).status_code == 200
    assert client.post("/ui/api/tools/mode", json={"tool": "shell", "mode": "sideways"}, headers=H).status_code == 400
    assert client.post("/ui/api/tools/default", json={"mode": "on_demand"}, headers=H).status_code == 200
    rid = client.post("/ui/api/tools/rules", json={"pattern": "read_*", "source": "workspace"}, headers=H).json()["id"]
    assert client.post("/ui/api/tools/source", json={"tool": "docs_search", "source": "docs"}, headers=H).status_code == 200
    lst = client.get("/ui/api/tools").json()
    assert {g["source"] for g in lst["sources"]} == {"client", "unity", "workspace", "docs"}
    assert client.post(f"/ui/api/tools/rules/{rid}/remove", headers=H).status_code == 200
    assert client.post("/ui/api/tools/forget", json={"tool": "shell"}, headers=H).json()["forgotten"] == 1
    assert client.post("/ui/api/tools/mode", json={"source": "unity", "mode": "automatic"}).status_code == 403
    # CLI against the same database
    import yaml
    from app import cli
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(json.loads(orch.config.model_dump_json(exclude={"source_path"}))))
    assert cli.main(["--config", str(p), "tools", "list"]) == 0
    out = capsys.readouterr().out
    assert "unity" in out and "unity__read_console" in out and "on_demand" in out
    assert cli.main(["--config", str(p), "tools", "set", "unity__execute_code", "off"]) == 0
    assert cli.main(["--config", str(p), "tools", "set", "--source", "unity", "automatic"]) == 0
    assert cli.main(["--config", str(p), "tools", "default", "automatic"]) == 0
    assert cli.main(["--config", str(p), "tools", "rule", "add", "gh_*", "github"]) == 0
    assert orch.tools_acl.mode_of("unity__execute_code") == ("off", "tool")
    assert orch.tools_acl.mode_of("unity__read_console") == ("automatic", "source")
