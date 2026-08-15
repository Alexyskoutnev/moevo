"""Base evaluator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Result of evaluating a single task."""

    task_id: str
    score: float  # 0.0 to 1.0
    max_score: float
    rubric_breakdown: dict[str, float] = field(default_factory=dict)
    response_text: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_score(self) -> float:
        if self.max_score == 0:
            return 0.0
        return self.score / self.max_score


class BaseEvaluator(ABC):
    """Abstract base for all benchmark evaluators."""

    @abstractmethod
    async def evaluate(
        self, task_id: str, prompt: str, response: str, reference: str, **kwargs: Any
    ) -> EvalResult:
        """Evaluate a single response against the reference/rubric."""
        ...

    @abstractmethod
    def name(self) -> str: ...
