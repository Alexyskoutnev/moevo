"""Bounded Docker workspace for account-based benchmark agents."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from moevo.codex.client import CodexError, run_codex
from moevo.codex.finance_pilot import write_json


@contextmanager
def task_container(workspace: Path, image: str, *, root_workspace: bool = False):
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    # This directory is a task-only scratch directory, not the repository.
    workspace.chmod(0o777)
    # Resolve locally; do not silently pull a changed tag during evaluation.
    inspected = subprocess.check_output(
        ["docker", "image", "inspect", image], text=True, timeout=30
    )
    image_info = json.loads(inspected)[0]
    image_id = image_info["Id"]
    container_id = subprocess.check_output(
        [
            "docker",
            "run",
            "--detach",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--memory=4g",
            "--cpus=2",
            "--pids-limit=256",
            "--user=1000:1000",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=512m,mode=1777",
            "--mount",
            f"type=bind,source={workspace},target=/workspace",
            "--mount",
            f"type=bind,source={workspace},target=/app",
            "--workdir=/workspace",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            *(["--mount", f"type=bind,source={workspace},target=/root"] if root_workspace else []),
            image_id,
            "sleep",
            "infinity",
        ],
        text=True,
        timeout=30,
    ).strip()
    try:
        yield container_id, image_id
    finally:
        subprocess.run(["docker", "rm", "--force", container_id], capture_output=True, timeout=30)


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
        response = run_codex(
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
        if not trace_path.exists() or not json.loads(trace_path.read_text()):
            raise CodexError("No task-container tool call ran; E2E validation incomplete")
        write_json(
            output / "runtime.json",
            {
                "image_id": image_id,
                "network": "none",
                "cpus": 2,
                "memory": "4g",
                "max_tool_calls": max_tools,
                "timeout_seconds": timeout,
            },
        )
        return response
