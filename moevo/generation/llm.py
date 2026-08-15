"""LLM calls via OpenAI-compatible SDK with multi-provider support."""

from __future__ import annotations

import asyncio
import logging
import os

import openai

logger = logging.getLogger("moevo.llm")


def _get_client(model: str) -> tuple[openai.OpenAI, str]:
    """Return (client, actual_model_name) based on model prefix.

    Supports:
        gemini/model-name  → Google AI via OpenAI-compat endpoint
        anthropic/model    → Anthropic via OpenAI-compat endpoint
        openai/model       → OpenAI (or just model name)
    """
    if model.startswith("gemini/"):
        actual = model[len("gemini/") :]
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ), actual

    if model.startswith("anthropic/"):
        actual = model[len("anthropic/") :]
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://api.anthropic.com/v1/",
        ), actual

    actual = model[len("openai/") :] if model.startswith("openai/") else model

    api_key = os.environ.get("OPENAI_API_KEY", "")
    return openai.OpenAI(api_key=api_key), actual


def _is_reasoning_model(model: str) -> bool:
    """Check if model uses reasoning tokens (o1/o3/o4/gpt-5)."""
    lower = model.lower()
    return any(lower.startswith(p) for p in ("o1", "o3", "o4", "gpt-5"))


async def generate(
    model: str,
    system: str,
    user: str,
    temperature: float = 1.0,
    max_tokens: int = 16384,
    retries: int = 2,
    retry_delay: float = 5.0,
    timeout: float = 120.0,
) -> str:
    """Generate text from an LLM. Returns the response text."""
    client, actual_model = _get_client(model)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    params: dict = {
        "model": actual_model,
        "messages": messages,
    }

    if _is_reasoning_model(actual_model):
        params["max_completion_tokens"] = max_tokens
    else:
        params["temperature"] = temperature
        params["max_tokens"] = max_tokens

    for attempt in range(retries + 1):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(client.chat.completions.create, **params),
                timeout=timeout,
            )
            text = response.choices[0].message.content or ""
            logger.debug("LLM response (%d chars) from %s", len(text), actual_model)
            return text

        except Exception as e:
            if attempt < retries:
                logger.warning(
                    "LLM attempt %d/%d failed: %s, retrying...", attempt + 1, retries + 1, e
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error("LLM call failed after %d attempts: %s", retries + 1, e)
                raise

    raise RuntimeError("Unreachable")
