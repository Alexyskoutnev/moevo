"""Gemini CLI agent backend — runs tasks via the Google Gemini CLI.

Uses -o json (single JSON object) for reliability. The stream-json format
can cause infinite retry loops on API 500 errors, leading to timeouts.

JSON format:
  {
    "session_id": "...",
    "response": "The file has been created.",
    "stats": {
      "tools": {"byName": {"write_file": {"count": 1, ...}}},
      "models": {"gemini-3-flash-preview": {"tokens": {...}}}
    }
  }
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from moevo.eval.agents.base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class GeminiCLIAgent(BaseAgent):
    """Runs tasks through Gemini CLI (npx @google/gemini-cli)."""

    def name(self) -> str:
        return "gemini-cli"

    async def _run(self, prompt: str, cwd: Path) -> AgentResult:
        bin_path = self._find_binary("@google/gemini-cli", "gemini")

        cwd = cwd.resolve()
        cwd.mkdir(parents=True, exist_ok=True)

        full_prompt = (
            f"{self._system_prompt}\n\n"
            f"Your working directory is: {cwd}\n"
            f"All files MUST be saved there.\n\n"
            f"{prompt}"
        )

        model = self._model or "gemini-2.5-pro"
        cmd = [
            *bin_path.split(),
            "-p",
            full_prompt,
            "--yolo",
            "-o",
            "json",
            "--model",
            model,
        ]

        stdout, stderr, rc = await self._run_subprocess(cmd, cwd)

        result = AgentResult(raw_output=stdout)

        if rc != 0 and not stdout.strip():
            result.error = f"Exit code {rc}: {stderr[:500]}"
            return result

        # Parse JSON output
        try:
            data = json.loads(stdout)
            result.response = data.get("response", "")

            # Extract tool usage stats
            stats = data.get("stats", {})
            tools = stats.get("tools", {})
            for tool_name, info in tools.get("byName", {}).items():
                result.tool_calls.append(
                    {
                        "tool": tool_name,
                        "input": f"count={info.get('count', 0)} success={info.get('success', 0)}",
                    }
                )
                result.messages.append(
                    {
                        "role": "assistant",
                        "type": "tool_use",
                        "tool": tool_name,
                        "count": info.get("count", 0),
                    }
                )

            # Extract token usage
            models = stats.get("models", {})
            for model_name, model_info in models.items():
                tokens = model_info.get("tokens", {})
                if tokens:
                    result.tool_calls.append(
                        {
                            "tool": "_usage",
                            "input": (
                                f"model={model_name} "
                                f"in={tokens.get('input', 0)} "
                                f"out={tokens.get('candidates', 0)} "
                                f"cached={tokens.get('cached', 0)}"
                            ),
                        }
                    )
                    result.messages.append(
                        {
                            "role": "system",
                            "type": "usage",
                            "model": model_name,
                            "input_tokens": tokens.get("input", 0),
                            "output_tokens": tokens.get("candidates", 0),
                        }
                    )

        except json.JSONDecodeError:
            # If JSON parsing fails, treat raw stdout as the response
            result.response = stdout

        return result
