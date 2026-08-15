"""Configuration for moevo evolution runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MoevoConfig:
    """All configuration for an evolution run."""

    # Required
    initial_program: str = ""
    evaluator_path: str = ""
    objectives: list[str] = field(default_factory=lambda: ["score"])

    # LLM
    model: str = "gemini/gemini-3-flash-preview"
    temperature: float = 0.7
    max_tokens: int = 32000

    # Evolution
    iterations: int = 50
    population_size: int = 10
    num_islands: int = 2
    migration_interval: int = 5
    migration_count: int = 2
    retry_attempts: int = 2
    fresh_start: bool = True
    max_code_lines: int = 10000

    # Adaptation
    ucb_constant: float = 1.41
    decay: float = 0.9
    intensity_min: float = 0.2
    intensity_max: float = 0.6

    # Pareto
    ref_point: list[float] | None = None

    # Output
    output_dir: str = "results/moevo_run"
    checkpoint_interval: int = 5

    # Prompt
    diff_mode: bool = True
    num_context_programs: int = 2

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for moevo."""
    p = argparse.ArgumentParser(
        prog="moevo",
        description="Multi-objective evolutionary code optimization",
    )
    p.add_argument("initial_program", help="Path to initial program")
    p.add_argument("evaluator", help="Path to evaluator module with evaluate()")
    p.add_argument(
        "--objectives", nargs="+", default=["score"], help="Objective metric names (all maximized)"
    )
    p.add_argument("--model", default="gemini/gemini-3-flash-preview")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--population-size", type=int, default=10)
    p.add_argument("--num-islands", type=int, default=2)
    p.add_argument("--migration-interval", type=int, default=5)
    p.add_argument("--migration-count", type=int, default=2)
    p.add_argument("--retry-attempts", type=int, default=1)
    p.add_argument("--ucb-constant", type=float, default=1.41)
    p.add_argument("--decay", type=float, default=0.9)
    p.add_argument("--intensity-min", type=float, default=0.2)
    p.add_argument("--intensity-max", type=float, default=0.6)
    p.add_argument(
        "--ref-point",
        nargs="+",
        type=float,
        default=None,
        help="Reference point for hypervolume (one per objective)",
    )
    p.add_argument("--output-dir", default="results/moevo_run")
    p.add_argument("--checkpoint-interval", type=int, default=5)
    p.add_argument("--diff-mode", action="store_true", default=True)
    p.add_argument("--no-diff-mode", dest="diff_mode", action="store_false")
    p.add_argument("--num-context-programs", type=int, default=2)
    return p


def config_from_args(args: argparse.Namespace) -> MoevoConfig:
    """Create MoevoConfig from parsed CLI arguments."""
    return MoevoConfig(
        initial_program=args.initial_program,
        evaluator_path=args.evaluator,
        objectives=args.objectives,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        iterations=args.iterations,
        population_size=args.population_size,
        num_islands=args.num_islands,
        migration_interval=args.migration_interval,
        migration_count=args.migration_count,
        retry_attempts=args.retry_attempts,
        ucb_constant=args.ucb_constant,
        decay=args.decay,
        intensity_min=args.intensity_min,
        intensity_max=args.intensity_max,
        ref_point=args.ref_point,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        diff_mode=args.diff_mode,
        num_context_programs=args.num_context_programs,
    )
