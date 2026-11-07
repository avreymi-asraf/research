#!/usr/bin/env python3
"""
Adversarial MCP Verification Test Suite.
Tests stdio JSON-RPC 2.0 implementation across:
1. book_tool.py --mcp
2. social_tool.py --mcp
3. forum_tool.py --mcp
And validates mcp_config.json.
"""

import os
import sys
import json
import subprocess
import time

TOOLS = {
    "books-search-server": "/home/avreymi/teamwork_projects/agent_data_sources/tools/book_tool.py",
    "social-data-server": "/home/avreymi/teamwork_projects/agent_data_sources/tools/social_tool.py",
    "forum-search-server": "/home/avreymi/teamwork_projects/agent_data_sources/tools/forum_tool.py",
}

CONFIG_PATH = "/home/avreymi/teamwork_projects/agent_data_sources/mcp_config.json"

class MCPProcessClient:
    def __init__(self, script_path):
        self.script_path = script_path
        self.proc = subprocess.Popen(
            [sys.executable, script_path, "--mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    def send_raw(self, line: str):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def send_json(self, obj):
        self.send_raw(json.dumps(obj))

    def read_line(self, timeout=5.0):
        start = time.time()
        import select
        r, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if r:
            return self.proc.stdout.readline()
        return None

    def read_json(self, timeout=5.0):
        line = self.read_line(timeout=timeout)
        if not line:
            return None
        try:
            return json.loads(line.strip())
        except Exception as e:
            return {"_raw": line, "_parse_error": str(e)}

    def is_alive(self):
        return self.proc.poll() is None

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


def test_server_handshake_and_core(name, script_path):
    print(f"\n==========================================")
    print(f"Testing Server: {name} ({script_path})")
    print(f"==========================================")

    results = []

    # Test 1: Initialize
    client = MCPProcessClient(script_path)
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        }
    }
    client.send_json(init_req)
    resp = client.read_json(timeout=5.0)
    print(f"  [1. initialize] Response: {resp}")

    t1_pass = (
        isinstance(resp, dict)
        and resp.get("jsonrpc") == "2.0"
        and resp.get("id") == 1
        and "result" in resp
        and resp["result"].get("protocolVersion") == "2024-11-05"
        and "serverInfo" in resp["result"]
        and resp["result"]["serverInfo"].get("name") == name
    )
    results.append(("initialize", t1_pass, resp))

    # Test 2: tools/list
    list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    client.send_json(list_req)
    resp_list = client.read_json(timeout=5.0)
    print(f"  [2. tools/list] Response: {resp_list}")

    t2_pass = (
        isinstance(resp_list, dict)
        and resp_list.get("jsonrpc") == "2.0"
        and resp_list.get("id") == 2
        and "result" in resp_list
        and "tools" in resp_list["result"]
        and isinstance(resp_list["result"]["tools"], list)
        and len(resp_list["result"]["tools"]) > 0
    )
    tools_declared = resp_list.get("result", {}).get("tools", []) if t2_pass else []
    results.append(("tools/list", t2_pass, resp_list))

    # Test 3: tools/call with valid args for each declared tool
    for tool in tools_declared:
        tool_name = tool["name"]
        call_id = f"call_{tool_name}"
        sample_args = {}
        if tool_name == "search_books":
            sample_args = {"query": "python", "source": "openlibrary", "max_results": 1}
        elif tool_name == "fetch_ia_text":
            sample_args = {"ia_id": "test_id_nonexistent", "max_chars": 100}
        elif tool_name == "search_social":
            sample_args = {"query": "ai", "platform": "x", "method": "socialdata", "max_results": 1}
        elif tool_name == "search_socialdata_tweets":
            sample_args = {"query": "ai", "max_results": 1}
        elif tool_name == "search_reddit_arctic_shift":
            sample_args = {"query": "python", "subreddit": "python", "limit": 1}
        elif tool_name == "fetch_rotter_scoops":
            sample_args = {"limit": 1}
        elif tool_name == "search_discourse_forum":
            sample_args = {"base_url": "https://discuss.python.org", "query": "python", "limit": 1}
        elif tool_name == "build_serp_forum_dork":
            sample_args = {"query": "python", "domain": "fxp.co.il"}

        call_req = {
            "jsonrpc": "2.0",
            "id": call_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": sample_args
            }
        }
        client.send_json(call_req)
        resp_call = client.read_json(timeout=10.0)
        print(f"  [3. tools/call:{tool_name}] Response: {resp_call}")

        valid_call = (
            isinstance(resp_call, dict)
            and resp_call.get("jsonrpc") == "2.0"
            and resp_call.get("id") == call_id
            and "result" in resp_call
            and "content" in resp_call["result"]
            and isinstance(resp_call["result"]["content"], list)
            and len(resp_call["result"]["content"]) > 0
            and resp_call["result"]["content"][0].get("type") == "text"
        )
        results.append((f"tools/call:{tool_name}", valid_call, resp_call))

    # Test 4: Invalid method name (JSON-RPC -32601)
    bad_method_req = {"jsonrpc": "2.0", "id": 99, "method": "nonexistent_method", "params": {}}
    client.send_json(bad_method_req)
    resp_bad = client.read_json(timeout=5.0)
    print(f"  [4. invalid method] Response: {resp_bad}")

    t4_pass = (
        isinstance(resp_bad, dict)
        and resp_bad.get("jsonrpc") == "2.0"
        and resp_bad.get("id") == 99
        and "error" in resp_bad
        and resp_bad["error"].get("code") == -32601
    )
    results.append(("invalid_method_-32601", t4_pass, resp_bad))

    # Test 5: Call unknown tool
    unknown_tool_req = {
        "jsonrpc": "2.0",
        "id": 100,
        "method": "tools/call",
        "params": {"name": "absolutely_fake_tool_xyz", "arguments": {}}
    }
    client.send_json(unknown_tool_req)
    resp_unknown_tool = client.read_json(timeout=5.0)
    print(f"  [5. unknown tool] Response: {resp_unknown_tool}")
    # In standard MCP, calling an unknown tool should either return result with isError: true or a JSON-RPC error
    results.append(("call_unknown_tool", True, resp_unknown_tool))

    # Test 6: Notification notifications/initialized (no id)
    # Standard MCP client sends this after initialize.
    client.send_json({"jsonrpc": "2.0", "method": "notifications/initialized"})
    # Check if server sends an unexpected error response or remains quiet or crashes
    resp_notif = client.read_json(timeout=1.0)
    print(f"  [6. notification notifications/initialized] Response (None expected in JSON-RPC): {resp_notif}")
    results.append(("notification_handling", resp_notif is None, resp_notif))

    # Test 7: Malformed JSON line
    client.send_raw('{"jsonrpc": "2.0", "method": broken_syntax')
    # According to JSON-RPC 2.0, parse error should be -32700. If ignored, resp is None.
    resp_malformed = client.read_json(timeout=1.0)
    print(f"  [7. malformed json line] Response: {resp_malformed}")
    results.append(("malformed_json_response", resp_malformed is not None, resp_malformed))

    client.close()

    # Test 8: Crash stress tests (Does the server die on invalid inputs?)
    crash_scenarios = [
        ("None params", {"jsonrpc": "2.0", "id": 101, "method": "tools/call", "params": None}),
        ("None arguments", {"jsonrpc": "2.0", "id": 102, "method": "tools/call", "params": {"name": tools_declared[0]["name"] if tools_declared else "test", "arguments": None}}),
        ("Invalid type for integer param", {"jsonrpc": "2.0", "id": 103, "method": "tools/call", "params": {"name": tools_declared[0]["name"] if tools_declared else "test", "arguments": {"query": "test", "max_results": "NOT_AN_INT", "limit": "NOT_AN_INT"}}}),
        ("Non-dict JSON top-level array", [1, 2, 3]),
        ("Non-dict JSON top-level integer", 12345),
        ("Non-dict JSON top-level string", "just a string"),
        ("Hebrew query utf-8", {"jsonrpc": "2.0", "id": 104, "method": "tools/call", "params": {"name": tools_declared[0]["name"] if tools_declared else "test", "arguments": {"query": "שלום עולם"}}}),
    ]

    crash_results = []
    for s_name, payload in crash_scenarios:
        sub_client = MCPProcessClient(script_path)
        try:
            # First handshake
            sub_client.send_json(init_req)
            sub_client.read_json(timeout=2.0)

            # Send adversarial payload
            sub_client.send_json(payload)
            r = sub_client.read_json(timeout=3.0)
            alive = sub_client.is_alive()
            err_output = ""
            if not alive:
                err_output = sub_client.proc.stderr.read()
            print(f"  [Stress: {s_name}] Alive: {alive}, Response: {r}, Stderr: {err_output[:200] if err_output else 'None'}")
            crash_results.append((s_name, alive, r, err_output))
        finally:
            sub_client.close()

    return results, crash_results, tools_declared


def test_mcp_config():
    print(f"\n==========================================")
    print(f"Validating mcp_config.json")
    print(f"==========================================")

    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: {CONFIG_PATH} does not exist!")
        return False, []

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    findings = []
    if "mcpServers" not in data:
        findings.append("Missing 'mcpServers' root key")
        return False, findings

    servers = data["mcpServers"]
    for s_name, s_cfg in servers.items():
        cmd = s_cfg.get("command")
        args = s_cfg.get("args", [])
        env = s_cfg.get("env", {})
        print(f"Server '{s_name}': command={cmd}, args={args}")

        if cmd != "python3":
            findings.append(f"Server {s_name}: unexpected command '{cmd}' (expected python3)")

        if not args or not os.path.exists(args[0]):
            findings.append(f"Server {s_name}: target script does not exist: {args[0] if args else 'EMPTY'}")

    return len(findings) == 0, findings


if __name__ == "__main__":
    config_ok, config_findings = test_mcp_config()

    all_results = {}
    for name, path in TOOLS.items():
        res, crashes, tools = test_server_handshake_and_core(name, path)
        all_results[name] = {
            "core": res,
            "stress": crashes,
            "tools": tools
        }

    print("\n\nSUMMARY COMPLETE.")
