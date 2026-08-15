"""Base agent interface for task execution."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result of running an agent on a single task."""

    response: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    raw_output: str = ""


class _TokenBucket:
    """Async token-bucket rate limiter.

    Allows up to *rate* calls per 60 seconds across concurrent tasks.
    Each call to acquire() blocks until a token is available.
    """

    def __init__(self, rpm: float) -> None:
        self._interval = 60.0 / rpm  # seconds between tokens
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                delay = self._next_allowed - now
                logger.debug("[rate-limit] Waiting %.1fs", delay)
                await asyncio.sleep(delay)
            self._next_allowed = max(now, self._next_allowed) + self._interval


class BaseAgent(ABC):
    """Abstract base for all agent backends (Claude Code, Gemini CLI, Codex).

    Args:
        rpm: Max requests per minute. 0 = unlimited (default).
             Each call to run() acquires a token before executing,
             so this works transparently with any concurrency level.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        max_turns: int = 10,
        model: str | None = None,
        rpm: float = 0,
    ) -> None:
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._max_turns = max_turns
        self._model = model
        self._rate_limiter: _TokenBucket | None = _TokenBucket(rpm) if rpm > 0 else None

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are a professional assistant completing real-world work tasks. "
            "Follow the instructions precisely. Produce the exact deliverables requested.\n\n"
            "CRITICAL FILE RULES — you MUST follow these:\n"
            "1. Save ALL files (scripts, deliverables, temp files) to the CURRENT WORKING DIRECTORY (./) only.\n"
            "2. NEVER write to /tmp/, ~/, /Users/, or any absolute path outside ./\n"
            "3. Use ONLY relative paths (e.g., ./output.docx, ./script.py, ./data.csv).\n"
            "4. When running scripts, run them from ./ (e.g., python ./script.py).\n"
            "5. All deliverables must end up in ./ when you are done.\n\n"
            "If reference files are mentioned but not found in the working directory, "
            "use reasonable placeholder data and focus on demonstrating the correct "
            "methodology and structure. Be thorough but concise."
        )

    async def run(self, prompt: str, cwd: Path) -> AgentResult:
        """Run the agent, respecting rate limits if configured."""
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        return await self._run(prompt, cwd)

    @abstractmethod
    async def _run(self, prompt: str, cwd: Path) -> AgentResult:
        """Subclass implementation — execute the agent on a prompt."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name."""
        ...

    def _find_binary(self, npm_package: str, binary_name: str) -> str:
        """Find a CLI binary, preferring global install over npx."""
        path = shutil.which(binary_name)
        if path:
            return path
        # Fall back to npx
        return f"npx {npm_package}"

    async def _run_subprocess(
        self,
        cmd: list[str],
        cwd: Path,
        timeout: int = 1800,
        env: dict[str, str] | None = None,
    ) -> tuple[str, str, int]:
        """Run a subprocess and return (stdout, stderr, returncode)."""
        logger.debug("[subprocess] %s", " ".join(cmd))
        base_env = env if env is not None else os.environ
        run_env = {**base_env, "MPLBACKEND": "Agg"}
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=run_env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return "", f"Process timed out after {timeout}s", -1

        return (
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
            proc.returncode or 0,
        )

    @staticmethod
    def _parse_jsonl(raw: str) -> list[dict[str, Any]]:
        """Parse newline-delimited JSON (JSONL) output."""
        events = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events
