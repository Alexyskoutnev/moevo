"""OpenAI Codex CLI agent backend - runs tasks via the Codex CLI.

Codex JSONL format:
  {"type":"thread.started","thread_id":"..."}
  {"type":"turn.started"}
  {"type":"item.completed","item":{"type":"reasoning","text":"..."}}
  {"type":"item.completed","item":{"type":"command_execution","command":"...","exit_code":0}}
  {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
  {"type":"turn.completed","usage":{...}}
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from moevo.eval.agents.base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CodexAgent(BaseAgent):
    """Runs tasks through OpenAI Codex CLI (npx @openai/codex)."""

    def name(self) -> str:
        return "codex"

    async def _run(self, prompt: str, cwd: Path) -> AgentResult:
        bin_path = self._find_binary("@openai/codex", "codex")

        # Ensure cwd is absolute (Codex needs it to exist)
        cwd = cwd.resolve()
        cwd.mkdir(parents=True, exist_ok=True)

        # Init a git repo if needed (Codex requires it)
        git_dir = cwd / ".git"
        if not git_dir.exists():
            proc = await asyncio.create_subprocess_exec(
                "git",
                "init",
                cwd=str(cwd),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

        # Codex CLI: exec for non-interactive, --full-auto for auto-approve,
        # --json for JSONL output
        full_prompt = f"{self._system_prompt}\n\nYour working directory is: {cwd}\nAll files MUST be saved there.\n\n{prompt}"

        model = self._model or "gpt-5.4"
        cmd = [
            *bin_path.split(),
            "exec",
            "--full-auto",
            "--json",
            "-m",
            model,
            "-c",
            "model_reasoning_effort=xhigh",
            "-C",
            str(cwd),
            full_prompt,
        ]

        stdout, stderr, rc = await self._run_subprocess(cmd, cwd)

        result = AgentResult(raw_output=stdout)

        if rc != 0 and not stdout.strip():
            result.error = f"Exit code {rc}: {stderr[:500]}"
            return result

        # Parse Codex JSONL output
        events = self._parse_jsonl(stdout)
        if events:
            result.response, result.tool_calls, result.messages = self._parse_codex_events(events)
        else:
            result.response = stdout

        return result

    @staticmethod
    def _parse_codex_events(events: list[dict]) -> tuple[str, list[dict], list[dict]]:
        """Parse Codex CLI JSONL events into response + tool calls + messages."""
        response_parts: list[str] = []
        tool_calls: list[dict] = []
        messages: list[dict] = []

        for event in events:
            event_type = event.get("type", "")

            if event_type == "item.completed":
                item = event.get("item", {})
                item_type = item.get("type", "")

                if item_type == "command_execution":
                    cmd = item.get("command", "")
                    output = item.get("aggregated_output", "")
                    exit_code = item.get("exit_code")
                    tool_calls.append(
                        {
                            "tool": "shell",
                            "input": cmd[:500],
                            "output": output[:500],
                            "exit_code": exit_code,
                        }
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "type": "command",
                            "command": cmd[:1000],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "type": "command_output",
                            "output": output[:2000],
                            "exit_code": exit_code,
                        }
                    )
                elif item_type == "agent_message":
                    text = item.get("text", "")
                    if text:
                        response_parts.append(text)
                        messages.append({"role": "assistant", "type": "text", "content": text})
                elif item_type == "reasoning":
                    text = item.get("text", "")
                    tool_calls.append({"tool": "_reasoning", "input": text[:500]})
                    messages.append(
                        {"role": "assistant", "type": "reasoning", "content": text[:2000]}
                    )

            elif event_type == "turn.completed":
                usage = event.get("usage", {})
                if usage:
                    tool_calls.append(
                        {
                            "tool": "_usage",
                            "input": f"in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)} cached={usage.get('cached_input_tokens', 0)}",
                        }
                    )
                    messages.append(
                        {
                            "role": "system",
                            "type": "usage",
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                            "cached_tokens": usage.get("cached_input_tokens", 0),
                        }
                    )

        return "\n".join(response_parts), tool_calls, messages
