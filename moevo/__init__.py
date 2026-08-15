"""moevo — Multi-Objective Evolutionary Code Optimization.

A lean framework for evolving code using LLM-based mutation with
NSGA-II Pareto selection via pymoo.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .controller import MoevoController
from .core import DiscoveryResult, EvalResult, MoevoConfig, Program


async def run_discovery(
    evaluator: str,
    initial_program: str,
    objectives: list[str] | None = None,
    model: str = "gemini/gemini-3-flash-preview",
    iterations: int = 50,
    **kwargs,
) -> DiscoveryResult:
    """Run multi-objective evolution and return the result.

    Args:
        evaluator: Path to evaluator module with evaluate(program_path) -> dict.
        initial_program: Path to initial seed program.
        objectives: Metric names to optimize (all maximized).
        model: LLM model identifier (prefix with gemini/, anthropic/, or openai/).
        iterations: Number of evolution iterations.
        **kwargs: Additional MoevoConfig fields.

    Returns:
        DiscoveryResult with pareto_front, best_program, etc.
    """
    config = MoevoConfig(
        evaluator_path=evaluator,
        initial_program=initial_program,
        objectives=objectives or ["score"],
        model=model,
        iterations=iterations,
        **kwargs,
    )
    controller = MoevoController(config)
    return await controller.run()


__all__ = [
    "MoevoConfig",
    "MoevoController",
    "DiscoveryResult",
    "EvalResult",
    "Program",
    "run_discovery",
]
