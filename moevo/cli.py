"""CLI entry point for moevo."""

from __future__ import annotations

import asyncio
import logging
import sys

from .controller import MoevoController
from .core import build_arg_parser, config_from_args


def main(argv: list[str] | None = None) -> int:
    """Run moevo from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    controller = MoevoController(config)
    result = asyncio.run(controller.run())

    # Summary
    print(f"\n{'=' * 60}")
    print(f"moevo complete: {result.iterations_completed} iterations")
    if result.task_evaluations is not None:
        print(f"Task evaluations: {result.task_evaluations}; stop: {result.stop_reason}")
    print(f"Pareto front: {len(result.pareto_front)} programs")
    print(f"Hypervolume: {result.hypervolume:.4f}")
    if result.best_program:
        scores = ", ".join(
            f"{o}={result.best_program.get_objective(o):.4f}" for o in config.objectives
        )
        print(f"Best program: {scores}")
    print(f"Output: {config.output_dir}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
