"""CustomHarnessAgent — production-ready general-purpose agent.

Subclasses BaseAgent for integration with the GDPval evaluation pipeline.
All logic is delegated to the loop, tools, prompt, and config modules.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from google import genai
from google.genai.types import GenerateContentConfig

from moevo.eval.agents.base import AgentResult, BaseAgent
from moevo.harness.config import (
    DEFAULT_MODEL,
    MAX_ITERATIONS,
    MAX_OUTPUT_TOKENS,
    TEMPERATURE,
)
from moevo.harness.loop import run_agent_loop
from moevo.harness.prompt import SYSTEM_PROMPT
from moevo.harness.tools import TOOL_DECLARATIONS

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CustomHarnessAgent(BaseAgent):
    """General-purpose Python-native agent using Gemini API with tool use.

    Provides 7 tools (bash, read_file, write_file, edit_file, list_dir, grep,
    glob) modeled after Claude Code and Codex CLI. Not biased toward any
    specific benchmark — works as a general-purpose agentic coding assistant.
    """

    def name(self) -> str:
        return "custom-harness"

    async def run(self, prompt: str, cwd: Path) -> AgentResult:
        """Run the agent. Never raises — all errors become partial results.

        Runs the synchronous Gemini API loop in a thread via asyncio.to_thread()
        so multiple tasks can execute concurrently with asyncio.gather().
        """
        try:
            cwd = cwd.resolve()
            cwd.mkdir(parents=True, exist_ok=True)

            # Initialize Gemini client
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                return AgentResult(error="GEMINI_API_KEY or GOOGLE_API_KEY required.")
            client = genai.Client(api_key=api_key)

            model = self._model or DEFAULT_MODEL
            max_iters = self._max_turns if self._max_turns != 10 else MAX_ITERATIONS
            system = SYSTEM_PROMPT.format(cwd=cwd)

            config = GenerateContentConfig(
                system_instruction=system,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                tools=[TOOL_DECLARATIONS],
            )

            # Run synchronous loop in a thread so asyncio.gather works concurrently
            response_text, tool_calls_log, messages_log = await asyncio.to_thread(
                run_agent_loop,
                client,
                model,
                config,
                prompt,
                cwd,
                max_iters,
            )

            return AgentResult(
                response=response_text,
                tool_calls=tool_calls_log,
                messages=messages_log,
            )
        except Exception as e:
            logger.error("[custom-harness] Unhandled error: %s: %s", type(e).__name__, e)
            return AgentResult(
                response=f"(Agent terminated due to error: {type(e).__name__}: {e})",
            )
