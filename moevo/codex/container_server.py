"""MCP terminal for a single isolated benchmark container; no host shell tool."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from moevo.codex.finance_pilot import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    config = json.loads(parser.parse_args().control.read_text())
    server = FastMCP("benchmark")
    lock = threading.Lock()
    trace: list[dict] = []

    @server.tool()
    def run(command: str, timeout_seconds: int = 60) -> str:
        """Run a shell command inside the task container at /workspace.

        Use this terminal to inspect inputs, write analysis/solution files, and
        run installed software. Network is unavailable. Timeout is 1–300 seconds.
        """
        if not 1 <= timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be 1–300")
        with lock:
            if len(trace) >= config["max_tool_calls"]:
                raise ValueError("Task tool-call budget exhausted")
            invocation = [
                "docker",
                "exec",
                "-w",
                "/workspace",
                config["container_id"],
                "timeout",
                "--kill-after=5",
                str(timeout_seconds),
                "bash",
                "-c",
                command,
            ]
            # The timeout runs inside the container so killing a docker client
            # cannot leave an unbounded task process behind.
            result = subprocess.run(
                invocation, capture_output=True, text=True, timeout=timeout_seconds + 20
            )
            record = {
                "command": command,
                "timeout_seconds": timeout_seconds,
                "exit_code": result.returncode,
                "stdout": result.stdout[-32000:],
                "stderr": result.stderr[-8000:],
            }
            trace.append(record)
            write_json(Path(config["trace_output"]), trace)
            return json.dumps(record)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
