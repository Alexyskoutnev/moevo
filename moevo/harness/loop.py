"""Core ReAct agent loop with retry, doom-loop detection, and safe response parsing.

This module is the engine — it calls the LLM, dispatches tool calls, manages
conversation history, detects stuck loops, and collects structured logs.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from google.genai.types import Content, GenerateContentConfig, Part

from moevo.harness.config import (
    API_BACKOFF_BASE,
    API_MAX_RETRIES,
    DOOM_LOOP_NUDGES,
    DOOM_LOOP_WINDOW,
)
from moevo.harness.tools import execute_tool

if TYPE_CHECKING:
    from pathlib import Path

    from google import genai

logger = logging.getLogger(__name__)


# ── Response parsing helpers ────────────────────────────────────────────────


def extract_function_calls(response: Any) -> list:
    """Extract FunctionCall objects from a Gemini response."""
    calls = []
    if not response.candidates:
        return calls
    content = response.candidates[0].content
    if not content or not content.parts:
        return calls
    for part in content.parts:
        if part.function_call:
            calls.append(part.function_call)
    return calls


def get_response_text(response: Any) -> str:
    """Safely extract text from a Gemini response."""
    try:
        return response.text or ""
    except (AttributeError, ValueError):
        if response.candidates:
            content = response.candidates[0].content
            if content and content.parts:
                texts = [p.text for p in content.parts if p.text]
                return "\n".join(texts)
        return ""


def get_content_parts(response: Any) -> list:
    """Safely get content parts from a Gemini response."""
    if not response.candidates:
        return []
    content = response.candidates[0].content
    if not content or not content.parts:
        return []
    return content.parts


# ── API call with retry ─────────────────────────────────────────────────────


def call_with_retry(
    client: genai.Client,
    model: str,
    contents: list,
    config: GenerateContentConfig,
) -> Any | None:
    """Call Gemini API with exponential backoff retry.

    Returns the response, or None if all retries fail.
    """
    for attempt in range(API_MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            error_name = type(e).__name__
            if attempt == API_MAX_RETRIES:
                logger.error(
                    "[harness] API failed after %d retries: %s: %s",
                    API_MAX_RETRIES,
                    error_name,
                    e,
                )
                return None
            wait = API_BACKOFF_BASE ** (attempt + 1)
            logger.warning(
                "[harness] API error (attempt %d): %s: %s — retry in %ds",
                attempt + 1,
                error_name,
                e,
                wait,
            )
            time.sleep(wait)
    return None


# ── Main agent loop ─────────────────────────────────────────────────────────


def run_agent_loop(
    client: genai.Client,
    model: str,
    config: GenerateContentConfig,
    prompt: str,
    cwd: Path,
    max_iterations: int,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute the ReAct tool-use loop.

    Never raises — all exceptions are caught and turned into partial results.

    Returns:
        (response_text, tool_calls_log, messages_log)
    """
    tool_calls_log: list[dict[str, Any]] = []
    messages_log: list[dict[str, Any]] = []
    response_parts: list[str] = []

    try:
        _run_loop_inner(
            client,
            model,
            config,
            prompt,
            cwd,
            max_iterations,
            tool_calls_log,
            messages_log,
            response_parts,
        )
    except Exception as e:
        logger.error("[harness] Unhandled error: %s: %s", type(e).__name__, e)
        response_parts.append(f"(Agent terminated due to error: {type(e).__name__}: {e})")

    return "\n".join(response_parts), tool_calls_log, messages_log


def _run_loop_inner(
    client: genai.Client,
    model: str,
    config: GenerateContentConfig,
    prompt: str,
    cwd: Path,
    max_iterations: int,
    tool_calls_log: list[dict[str, Any]],
    messages_log: list[dict[str, Any]],
    response_parts: list[str],
) -> None:
    """Inner loop logic — may raise; caller catches everything."""
    contents: list[Content | str] = [prompt]
    last_calls: list[str] = []
    nudge_count = 0

    for iteration in range(max_iterations):
        logger.debug("[harness] Iteration %d/%d", iteration + 1, max_iterations)

        response = call_with_retry(client, model, contents, config)
        if response is None:
            response_parts.append("I was unable to complete the task due to API errors.")
            break

        function_calls = extract_function_calls(response)

        # ── No tool calls → model is done ──
        if not function_calls:
            text = get_response_text(response)
            response_parts.append(text)
            messages_log.append(
                {
                    "role": "assistant",
                    "type": "text",
                    "content": text[:2000],
                }
            )
            break

        # ── Log any interleaved text ──
        for part in get_content_parts(response):
            if part.text:
                response_parts.append(part.text)
                messages_log.append(
                    {
                        "role": "assistant",
                        "type": "text",
                        "content": part.text[:2000],
                    }
                )

        # ── Execute tool calls ──
        tool_response_parts: list[Part] = []
        for fc in function_calls:
            try:
                args = dict(fc.args) if fc.args else {}
            except Exception:
                args = {}
            result = execute_tool(fc.name, args, cwd)

            tool_calls_log.append(
                {
                    "tool": fc.name,
                    "input": str(args)[:500],
                }
            )
            messages_log.append(
                {
                    "role": "assistant",
                    "type": "tool_use",
                    "tool": fc.name,
                    "input": str(args)[:1000],
                }
            )
            messages_log.append(
                {
                    "role": "tool",
                    "type": "tool_result",
                    "content": result[:2000],
                }
            )

            tool_response_parts.append(
                Part.from_function_response(
                    name=fc.name,
                    response={"result": result},
                )
            )

            call_sig = f"{fc.name}:{hash(str(args))}"
            last_calls.append(call_sig)

        # ── Append to conversation history ──
        try:
            model_content = response.candidates[0].content if response.candidates else None
        except (IndexError, AttributeError):
            model_content = None
        if model_content:
            contents.append(model_content)
        contents.append(Content(parts=tool_response_parts))

        # ── Doom-loop detection ──
        if len(last_calls) >= DOOM_LOOP_WINDOW:
            half = DOOM_LOOP_WINDOW // 2
            if last_calls[-half:] == last_calls[-DOOM_LOOP_WINDOW:-half]:
                nudge_count += 1
                if nudge_count >= DOOM_LOOP_NUDGES:
                    response_parts.append("(Agent loop terminated: stuck in repeated actions)")
                    break
                contents.append(
                    "You seem stuck repeating the same actions. "
                    "Try a different approach or finish with what you have."
                )
