"""Core data types for moevo."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Program:
    """A candidate program in the evolutionary population."""

    id: str
    solution: str
    metrics: dict[str, float] = field(default_factory=dict)
    parent_id: str | None = None
    island_id: int = 0
    iteration: int = 0
    feedback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_objective(self, name: str) -> float:
        value = self.metrics[name]
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"Objective {name!r} must be a finite number, got {value!r}")
        return float(value)

    def get_objectives(self, names: list[str]) -> list[float]:
        return [self.get_objective(n) for n in names]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "solution": self.solution,
            "metrics": self.metrics,
            "parent_id": self.parent_id,
            "island_id": self.island_id,
            "iteration": self.iteration,
            "feedback": self.feedback,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Program:
        return cls(
            id=d["id"],
            solution=d["solution"],
            metrics=d.get("metrics", {}),
            parent_id=d.get("parent_id"),
            island_id=d.get("island_id", 0),
            iteration=d.get("iteration", 0),
            feedback=d.get("feedback", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class EvalResult:
    """Result from evaluating a program."""

    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class DiscoveryResult:
    """Final result of an evolution run."""

    pareto_front: list[Program]
    best_program: Program | None
    all_programs: list[Program]
    iterations_completed: int
    hypervolume: float = 0.0
    stop_reason: str = "iterations_completed"
    task_evaluations: int | None = None
