"""Versioned task runtime: successful direct answers may use zero tool calls.

Container permissions, tools and account transport match container_runtime.
Tool use is an observation, not an eligibility rule for native task grading.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from moevo.codex import client
from moevo.codex.container_runtime import task_container
from moevo.codex.finance_pilot import write_json

PROTOCOL = "native-grading-with-optional-tool-use-v2"


def solve_in_container(
    prompt: str,
    workspace: Path,
    output: Path,
    *,
    image: str = "docker.io/library/moevo-or-runtime:20260915",
    timeout: int = 1200,
    max_tools: int = 80,
    root_workspace: bool = False,
):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (
        task_container(workspace, image, root_workspace=root_workspace) as (container_id, image_id),
        tempfile.TemporaryDirectory(prefix="moevo-container-agent-") as cli_cwd,
        tempfile.TemporaryDirectory(prefix="moevo-container-control-") as control,
    ):
        config = Path(control) / "control.json"
        write_json(
            config,
            {
                "container_id": container_id,
                "trace_output": str(output / "tool_trace.json"),
                "max_tool_calls": max_tools,
            },
        )
        response = client.run_codex(
            prompt,
            cwd=Path(cli_cwd),
            timeout=timeout,
            tools=False,
            mcp_server=[
                sys.executable,
                "-m",
                "moevo.codex.container_server",
                "--control",
                str(config),
            ],
            approved_mcp_tools=("run",),
            log_dir=output / "codex",
        )
        trace_path = output / "tool_trace.json"
        trace = json.loads(trace_path.read_text()) if trace_path.exists() else []
        if not isinstance(trace, list):
            raise ValueError("Task-container trace is malformed")
        # Do not fabricate a trace file or successful tool invocation. A CLI/MCP
        # transport failure still raises in client.run_codex before this point.
        write_json(
            output / "runtime.json",
            {
                "protocol": PROTOCOL,
                "image_id": image_id,
                "network": "none",
                "cpus": 2,
                "memory": "4g",
                "max_tool_calls": max_tools,
                "timeout_seconds": timeout,
                "tool_calls": len(trace),
                "direct_response": not trace,
                "tool_trace_present": trace_path.exists(),
                "native_grade_pending": True,
            },
        )
        return response
