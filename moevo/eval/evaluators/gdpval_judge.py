"""GDPval evaluator using an LLM judge (OpenAI or Gemini).

Sends the task prompt, the agent's response, and the rubric to the judge,
which scores each criterion individually.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from moevo.eval.evaluators.base import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gpt-5.4"

JUDGE_SYSTEM_PROMPT = """\
You are a strict, impartial grading judge for professional work tasks.

You will receive:
1. TASK: The original task prompt given to an AI assistant.
2. RESPONSE: The assistant's full response (text output and tool usage).
3. RUBRIC: A list of scoring criteria in [+N] / [-N] format.

Your job is to evaluate the RESPONSE against each rubric criterion and assign points.

Rules:
- Score each criterion independently.
- Award full points ONLY if the criterion is clearly and fully met.
- Award 0 if the criterion is not met or only partially met.
- For negative criteria ([-N]), apply the penalty if the condition is true.
- Base your judgment on what the response actually demonstrates, not what it claims.
- If the task required file deliverables (Excel, Word, etc.) and the response only describes
  them in text without creating them, criteria about file format/structure are NOT met.
- Be strict: vague or partial matches do not earn points.

Output your evaluation as JSON with this exact structure:
{
  "criteria": [
    {
      "criterion": "[+2] The exact criterion text",
      "points_awarded": 2,
      "points_possible": 2,
      "reasoning": "Brief explanation of why points were awarded or not"
    }
  ],
  "total_score": 16,
  "max_score": 20,
  "overall_notes": "Brief overall assessment"
}

Return ONLY the JSON object, no other text."""


class GDPvalJudgeEvaluator(BaseEvaluator):
    """Scores GDPval tasks using an LLM judge.

    Supports OpenAI (gpt-5.4, etc.) and Gemini (gemini-3.1-pro, etc.)
    based on the model name prefix. Uses async clients for non-blocking
    concurrent scoring.
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        temperature: float = 0.0,
        max_retries: int = 5,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._backend = "openai" if model.startswith("gpt") or model.startswith("o") else "gemini"

        if self._backend == "openai":
            from openai import AsyncOpenAI

            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY required for OpenAI judge models.")
            self._openai_client = AsyncOpenAI(api_key=key)
        else:
            from google import genai

            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ValueError("GEMINI_API_KEY required for Gemini judge models.")
            self._gemini_client = genai.Client(api_key=key)

    def name(self) -> str:
        return f"gdpval_judge ({self._model})"

    async def evaluate(
        self,
        task_id: str,
        prompt: str,
        response: str,
        reference: str,
        **kwargs: Any,
    ) -> EvalResult:
        if not response.strip():
            # Parse max_score from rubric so this counts as a real 0% rather
            # than being invisible (max_score=0) in aggregate metrics.
            rubric_max = self._estimate_max_from_rubric(reference)
            return EvalResult(
                task_id=task_id,
                score=0.0,
                max_score=rubric_max,
                error="Empty response",
            )

        judge_prompt = self._build_judge_prompt(prompt, response, reference)

        for attempt in range(self._max_retries + 1):
            try:
                judge_response = await self._call_llm(judge_prompt)
                return self._parse_judge_response(task_id, judge_response, response)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    "Judge parse error on task %s (attempt %d): %s", task_id, attempt + 1, e
                )
                if attempt == self._max_retries:
                    return EvalResult(
                        task_id=task_id,
                        score=0.0,
                        max_score=0.0,
                        response_text=response[:2000],
                        error=f"Judge parse failed after {self._max_retries + 1} attempts: {e}",
                        metadata={"raw_judge_response": judge_response[:2000]},
                    )
            except Exception as e:
                logger.error("Judge API error on task %s: %s", task_id, e)
                if attempt == self._max_retries:
                    return EvalResult(
                        task_id=task_id,
                        score=0.0,
                        max_score=0.0,
                        response_text=response[:2000],
                        error=f"Judge API error: {e}",
                    )
                wait = min(2**attempt * 3, 30)
                logger.info("[judge] Retry %d in %.0fs...", attempt + 1, wait)
                await asyncio.sleep(wait)

        return EvalResult(task_id=task_id, score=0.0, max_score=0.0, error="Unexpected")

    @staticmethod
    def _estimate_max_from_rubric(rubric: str) -> float:
        """Sum positive [+N] criteria in the rubric to estimate max_score."""
        total = 0.0
        for m in re.finditer(r"\[\+(\d+(?:\.\d+)?)\]", rubric):
            total += float(m.group(1))
        return total if total > 0 else 1.0  # fallback to 1.0 to avoid div-by-zero

    def _build_judge_prompt(self, task_prompt: str, response: str, rubric: str) -> str:
        return f"""## TASK
{task_prompt}

## RESPONSE
{response}

## RUBRIC
{rubric}

Evaluate the RESPONSE against each criterion in the RUBRIC. Return JSON only."""

    async def _call_llm(self, prompt: str) -> str:
        if self._backend == "openai":
            return await self._call_openai(prompt)
        return await self._call_gemini(prompt)

    async def _call_openai(self, prompt: str) -> str:
        response = await self._openai_client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _call_gemini(self, prompt: str) -> str:
        from google.genai.types import GenerateContentConfig

        config = GenerateContentConfig(
            temperature=self._temperature,
            system_instruction=JUDGE_SYSTEM_PROMPT,
            response_mime_type="application/json",
        )
        # Run sync Gemini client in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._gemini_client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            ),
        )
        return response.text or ""

    def _parse_judge_response(
        self,
        task_id: str,
        raw_response: str,
        agent_response: str,
    ) -> EvalResult:
        cleaned = raw_response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)

        total_score = float(data["total_score"])
        max_score = float(data["max_score"])

        # Clamp score to [0, max_score] - judge penalty criteria can
        # push the raw total negative, which is not meaningful.
        if max_score > 0:
            total_score = max(0.0, min(total_score, max_score))

        breakdown: dict[str, float] = {}
        criteria_details: list[dict[str, Any]] = []
        for c in data.get("criteria", []):
            criterion_text = c.get("criterion", "unknown")[:80]
            points = float(c.get("points_awarded", 0))
            breakdown[criterion_text] = points
            criteria_details.append(
                {
                    "criterion": c.get("criterion", ""),
                    "points_awarded": points,
                    "points_possible": float(c.get("points_possible", 0)),
                    "reasoning": c.get("reasoning", ""),
                }
            )

        return EvalResult(
            task_id=task_id,
            score=total_score,
            max_score=max_score,
            rubric_breakdown=breakdown,
            response_text=agent_response[:2000],
            metadata={
                "num_criteria": len(data.get("criteria", [])),
                "criteria_met": sum(
                    1 for c in data.get("criteria", []) if float(c.get("points_awarded", 0)) > 0
                ),
                "overall_notes": data.get("overall_notes", ""),
                "criteria_details": criteria_details,
                "judge_model": self._model,
            },
        )
