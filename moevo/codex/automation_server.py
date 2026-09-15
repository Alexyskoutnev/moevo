"""Task-local MCP tools for simulated AutomationBench. No real SaaS or model APIs."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from moevo.codex.automation_adapter import load_official, make_state
from moevo.codex.finance_pilot import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.control.read_text())
    official = load_official(Path(config["source"]))
    state = make_state(config["task"], official)
    server = FastMCP("benchmark")
    lock = threading.Lock()
    calls = 0
    trace = []

    def persist():
        write_json(
            Path(config["state_output"]),
            {"world": state["world"].model_dump(mode="json"), "trace": trace, "tool_calls": calls},
        )

    def call(name, arguments):
        nonlocal calls
        with lock:
            if calls >= config["max_tool_calls"]:
                raise ValueError("Task tool budget exhausted")
            calls += 1
            try:
                if name == "search":
                    result = official["search"](**arguments)
                else:
                    # Match upstream normalization of optional empty objects.
                    values = json.loads(arguments["arguments"])
                    values = {
                        k: v for k, v in values.items() if not (isinstance(v, dict) and not v)
                    }
                    result = official["execute"](
                        world=state["world"],
                        tool_name=arguments["tool_name"],
                        arguments=json.dumps(values),
                    )
                trace.append({"tool": name, "arguments": arguments, "result": result})
                return result
            except Exception as exc:
                trace.append({"tool": name, "arguments": arguments, "error": str(exc)})
                raise
            finally:
                persist()

    @server.tool()
    def search_tools(query: str, top_k: int = 5) -> str:
        """Discover simulated business tools by service, action, or keywords; returns parameter schemas."""
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be 1–20")
        return call("search", {"query": query, "top_k": top_k})

    @server.tool()
    def execute_tool(tool_name: str, arguments: str) -> str:
        """Execute a discovered simulated business tool; arguments is a JSON object encoded as a string."""
        return call("execute", {"tool_name": tool_name, "arguments": arguments})

    persist()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
