"""GDPval evaluator - keyword/rubric-based scoring.

This is a lightweight programmatic evaluator that checks rubric criteria
against the agent's response text. For production use, this should be
replaced or augmented with a frozen judge model (e.g., Gemini 3 Pro).
"""

from __future__ import annotations

import re
from typing import Any

from moevo.eval.evaluators.base import BaseEvaluator, EvalResult


class GDPvalEvaluator(BaseEvaluator):
    """Scores GDPval tasks using rubric criteria.

    Parses the rubric_pretty format where each criterion is:
        [+N] Description of what should be present.

    Checks whether the agent response contains evidence of meeting
    each criterion via keyword matching. This is a baseline heuristic;
    a frozen judge model should be used for accurate scoring.
    """

    def name(self) -> str:
        return "gdpval"

    async def evaluate(
        self,
        task_id: str,
        prompt: str,
        response: str,
        reference: str,
        **kwargs: Any,
    ) -> EvalResult:
        """Evaluate a GDPval task response against rubric criteria.

        Args:
            task_id: Unique task identifier.
            prompt: The original task prompt.
            response: The agent's full response text.
            reference: The rubric_pretty string with [+N] criteria.
            **kwargs: Additional metadata (rubric_json, etc.).
        """
        criteria = self._parse_rubric(reference)
        if not criteria:
            return EvalResult(
                task_id=task_id,
                score=0.0,
                max_score=0.0,
                response_text=response,
                error="No rubric criteria found",
            )

        total_points = sum(c["points"] for c in criteria)
        earned_points = 0.0
        breakdown: dict[str, float] = {}

        response_lower = response.lower()

        for c in criteria:
            desc = c["description"]
            key = desc[:80]
            # Extract key terms from the criterion description
            keywords = self._extract_keywords(desc)
            # Check if response addresses this criterion
            matched = self._check_criterion(response_lower, keywords)
            points = c["points"] if matched else 0.0
            earned_points += points
            breakdown[key] = points

        return EvalResult(
            task_id=task_id,
            score=earned_points,
            max_score=total_points,
            rubric_breakdown=breakdown,
            response_text=response[:2000],
            metadata={
                "num_criteria": len(criteria),
                "criteria_met": sum(1 for v in breakdown.values() if v > 0),
            },
        )

    def _parse_rubric(self, rubric_pretty: str) -> list[dict[str, Any]]:
        """Parse [+N] criterion lines from rubric_pretty."""
        criteria = []
        pattern = r"\[([+-]\d+)\]\s*(.+?)(?=\[([+-]\d+)\]|$)"
        for match in re.finditer(pattern, rubric_pretty, re.DOTALL):
            points = int(match.group(1))
            description = match.group(2).strip()
            criteria.append({"points": points, "description": description})
        return criteria

    def _extract_keywords(self, description: str) -> list[str]:
        """Extract meaningful keywords from a criterion description."""
        # Remove common filler words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "that",
            "this",
            "these",
            "those",
            "and",
            "or",
            "but",
            "if",
            "while",
            "because",
            "until",
            "so",
            "than",
            "too",
            "very",
            "just",
            "about",
            "each",
            "every",
            "all",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "any",
        }
        # Also extract quoted strings as exact-match terms
        quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", description)
        exact_terms = [q[0] or q[1] for q in quoted]

        words = re.findall(r"\b[a-z][a-z0-9_-]{2,}\b", description.lower())
        keywords = [w for w in words if w not in stop_words]
        return exact_terms + keywords[:10]

    def _check_criterion(self, response_lower: str, keywords: list[str]) -> bool:
        """Check if response likely addresses a criterion based on keywords."""
        if not keywords:
            return False
        # Require at least 30% of keywords to be present
        threshold = max(1, len(keywords) * 0.3)
        matches = sum(1 for kw in keywords if kw.lower() in response_lower)
        return matches >= threshold
