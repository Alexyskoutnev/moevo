"""Claude Code agent backend - runs tasks via the Claude Code CLI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from moevo.eval.agents.base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class ClaudeCodeAgent(BaseAgent):
    """Runs tasks through Claude Code CLI (npx @anthropic-ai/claude-code).

    Uses --output-format stream-json to capture the full message stream
    including tool use, tool results, and assistant text.
    """

    def name(self) -> str:
        return "claude-code"

    async def _run(self, prompt: str, cwd: Path) -> AgentResult:
        import os

        cwd = cwd.resolve()
        cwd.mkdir(parents=True, exist_ok=True)

        bin_path = self._find_binary("@anthropic-ai/claude-code", "claude")

        # Inject the absolute workspace path into the system prompt
        sys_prompt = (
            self._system_prompt
            + f"\n\nYour working directory is: {cwd}\nAll files MUST be saved there."
        )

        model = self._model or "claude-opus-4-6"
        cmd = [
            *bin_path.split(),
            "--model",
            model,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--system-prompt",
            sys_prompt,
            "--max-turns",
            str(self._max_turns),
            prompt,
        ]

        # Build a clean env without CLAUDE* vars so subprocess doesn't think it's nested
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
        stdout, stderr, rc = await self._run_subprocess(cmd, cwd, env=clean_env)

        result = AgentResult(raw_output=stdout)

        if rc != 0 and not stdout.strip():
            result.error = f"Exit code {rc}: {stderr[:500]}"
            return result

        # Parse stream-json JSONL output
        events = self._parse_jsonl(stdout)
        if events:
            result.response, result.tool_calls, result.messages = self._parse_claude_events(events)
        else:
            result.response = stdout

        return result

    @staticmethod
    def _parse_claude_events(events: list[dict[str, Any]]) -> tuple[str, list[dict], list[dict]]:
        """Parse Claude Code stream-json JSONL into response + tool calls + messages."""
        response_parts: list[str] = []
        tool_calls: list[dict] = []
        messages: list[dict] = []

        for event in events:
            msg_type = event.get("type", "")

            if msg_type == "assistant":
                message = event.get("message", {})
                for block in message.get("content", []):
                    if block.get("type") == "text":
                        text = block["text"]
                        response_parts.append(text)
                        messages.append({"role": "assistant", "type": "text", "content": text})
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "tool": block.get("name", ""),
                                "input": str(block.get("input", ""))[:500],
                            }
                        )
                        messages.append(
                            {
                                "role": "assistant",
                                "type": "tool_use",
                                "tool": block.get("name", ""),
                                "input": str(block.get("input", ""))[:1000],
                            }
                        )

            elif msg_type == "user":
                message = event.get("message", {})
                for block in message.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        content = block.get("content", "")
                        if isinstance(content, str) and content:
                            tool_calls.append({"tool": "_result", "input": content[:500]})
                            messages.append(
                                {
                                    "role": "tool",
                                    "type": "tool_result",
                                    "content": content[:2000],
                                }
                            )

            elif msg_type == "result":
                result_text = event.get("result", "")
                if result_text:
                    response_parts.append(result_text)
                messages.append(
                    {
                        "role": "system",
                        "type": "result",
                        "duration_ms": event.get("duration_ms"),
                        "num_turns": event.get("num_turns"),
                        "cost_usd": event.get("total_cost_usd"),
                    }
                )

        return "\n".join(response_parts), tool_calls, messages
