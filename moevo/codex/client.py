"""Call the installed Codex CLI using ChatGPT authentication, never an API key."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "gpt-6-astra"


class CodexError(RuntimeError):
    """A CLI call failed; do not treat it as a benchmark result."""


@dataclass
class CodexResponse:
    text: str
    events: list[dict] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0


def account_environment() -> dict[str, str]:
    """Keep the user's login, stripping environment-based API authentication."""
    blocked = {"CODEX_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"}
    return {k: v for k, v in os.environ.items() if not k.endswith("API_KEY") and k not in blocked}


def require_chatgpt_login() -> str:
    result = subprocess.run(
        ["codex", "login", "status"],
        capture_output=True,
        text=True,
        env=account_environment(),
        timeout=15,
    )
    status = result.stdout + result.stderr
    if result.returncode or "Logged in using ChatGPT" not in status:
        raise CodexError(
            "Codex must be signed in using ChatGPT. Run codex login; API fallback is disabled."
        )
    return subprocess.check_output(
        ["codex", "--version"], text=True, env=account_environment()
    ).strip()


def build_command(
    cwd: Path,
    output: Path,
    model: str,
    effort: str,
    tools: bool,
    schema_path: Path | None = None,
    mcp_server: list[str] | None = None,
    approved_mcp_tools: tuple[str, ...] = (),
) -> list[str]:
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Unsupported Astra reasoning effort")
    command = [
        "codex",
        "exec",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--ephemeral",
        "--json",
        "--color",
        "never",
        "--sandbox",
        "workspace-write" if tools else "read-only",
        "-C",
        str(cwd),
        "-m",
        model,
        "-c",
        'forced_login_method="chatgpt"',
        "-c",
        f'model_reasoning_effort="{effort}"',
        "-c",
        'web_search="disabled"',
        "-c",
        'approval_policy="never"',
        "-c",
        "sandbox_workspace_write.network_access=false",
        "-c",
        f"features.shell_tool={str(tools).lower()}",
        "--output-last-message",
        str(output),
    ]
    if schema_path is not None:
        command.extend(["--output-schema", str(schema_path)])
    if mcp_server:
        for key, value in {
            "command": mcp_server[0],
            "args": mcp_server[1:],
            "required": True,
            "startup_timeout_sec": 60,
            "tool_timeout_sec": 360,
            "default_tools_approval_mode": "auto",
        }.items():
            command.extend(["-c", f"mcp_servers.benchmark.{key}={json.dumps(value)}"])
        # Approval is restricted to the caller's known, task-local tools. This
        # does not change the user's config or authorize unrelated MCP servers.
        for name in approved_mcp_tools:
            if not name.isidentifier():
                raise ValueError("MCP tool names must be identifiers")
            command.extend(["-c", f'mcp_servers.benchmark.tools.{name}.approval_mode="approve"'])
    return [*command, "-"]


def run_codex(
    prompt: str,
    *,
    cwd: Path,
    model: str = DEFAULT_MODEL,
    effort: str = "xhigh",
    tools: bool = False,
    timeout: float = 300,
    schema: dict | None = None,
    log_dir: Path | None = None,
    mcp_server: list[str] | None = None,
    approved_mcp_tools: tuple[str, ...] = (),
) -> CodexResponse:
    cwd = cwd.resolve()
    cwd.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="moevo-codex-control-") as control:
        output = Path(control) / "response.txt"
        schema_path = None
        if schema is not None:
            schema_path = Path(control) / "schema.json"
            schema_path.write_text(json.dumps(schema))
        command = build_command(
            cwd, output, model, effort, tools, schema_path, mcp_server, approved_mcp_tools
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=account_environment(),
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(prompt, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
            raise CodexError(f"Codex exceeded {timeout}s; its process group was stopped") from exc
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "events.jsonl").write_text(stdout)
            (log_dir / "stderr.txt").write_text(stderr)
        events = []
        for line in stdout.splitlines():
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    events.append(item)
            except json.JSONDecodeError:
                continue
        failures = [e for e in events if e.get("type") in {"error", "turn.failed"}]
        completed = [e for e in events if e.get("type") == "turn.completed"]
        if process.returncode or failures or not completed or not output.exists():
            detail = json.dumps(failures[-1]) if failures else stderr[-1500:]
            raise CodexError(f"Codex failed (exit {process.returncode}): {detail}")
        usage: dict[str, int] = {}
        for event in completed:
            for key, value in event.get("usage", {}).items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
        return CodexResponse(output.read_text(), events, usage, time.monotonic() - start)
