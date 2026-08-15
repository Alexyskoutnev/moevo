"""Main evolution loop for moevo."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from .core import DiscoveryResult, MoevoConfig, Program
from .core.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from .generation import build_prompt, generate, load_evaluate_fn, parse_response, run_evaluation
from .search import ParetoDatabase

logger = logging.getLogger("moevo.controller")


class MoevoController:
    """Orchestrates the evolution loop."""

    def __init__(self, config: MoevoConfig):
        self.config = config
        self._evaluate_fn = None
        self._db: ParetoDatabase | None = None

    async def run(self) -> DiscoveryResult:
        """Run the full evolution loop."""
        cfg = self.config
        output_dir = cfg.output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load evaluator
        self._evaluate_fn = load_evaluate_fn(cfg.evaluator_path)

        # Initialize fresh (don't resume from stale checkpoints in cascade mode —
        # each slice should start clean with its own seed evaluation)
        start_iteration = 0
        if not cfg.fresh_start:
            latest = find_latest_checkpoint(output_dir)
            if latest:
                self._db, start_iteration = load_checkpoint(latest)
                start_iteration += 1
                logger.info("Resuming from iteration %d", start_iteration)
        if start_iteration == 0:
            self._db = ParetoDatabase(
                objectives=cfg.objectives,
                population_size=cfg.population_size,
                num_islands=cfg.num_islands,
                migration_interval=cfg.migration_interval,
                migration_count=cfg.migration_count,
                ref_point=cfg.ref_point,
            )
            # Seed with initial program
            await self._seed_initial_program()

        # Main loop
        for iteration in range(start_iteration, cfg.iterations):
            logger.info("=== Iteration %d/%d ===", iteration + 1, cfg.iterations)
            await self._run_iteration(iteration)

            self._db.end_iteration(iteration)

            # Log status
            front = self._db.get_pareto_front()
            hv = self._db.get_hypervolume()
            logger.info("Pareto front: %d programs, HV=%.4f", len(front), hv)
            for p in front:
                scores = ", ".join(f"{o}={p.get_objective(o):.4f}" for o in cfg.objectives)
                logger.info("  %s: %s", p.id[:8], scores)

            # Checkpoint
            if (iteration + 1) % cfg.checkpoint_interval == 0:
                save_checkpoint(self._db, iteration, output_dir)

        # Final checkpoint
        save_checkpoint(self._db, cfg.iterations - 1, output_dir)

        # Save best program
        best = self._db.get_best_program()
        if best:
            best_path = output_dir / "best_program.py"
            best_path.write_text(best.solution)
            logger.info("Best program saved to %s", best_path)

        # Save Pareto front programs
        front = self._db.get_pareto_front()
        front_dir = output_dir / "pareto_front"
        front_dir.mkdir(exist_ok=True)
        for i, p in enumerate(front):
            (front_dir / f"program_{i:02d}_{p.id[:8]}.py").write_text(p.solution)

        return DiscoveryResult(
            pareto_front=front,
            best_program=best,
            all_programs=self._db.all_programs,
            iterations_completed=cfg.iterations,
            hypervolume=self._db.get_hypervolume(),
        )

    async def _seed_initial_program(self) -> None:
        """Evaluate and add the initial seed program."""
        cfg = self.config
        code = Path(cfg.initial_program).read_text()
        program_id = _make_id()

        logger.info("Evaluating seed program...")
        result = await run_evaluation(self._evaluate_fn, code, program_id)

        if result.error:
            logger.warning("Seed evaluation failed: %s", result.error)

        seed = Program(
            id=program_id,
            solution=code,
            metrics=result.metrics,
            island_id=0,
            iteration=0,
            feedback=result.artifacts.get("feedback", ""),
        )

        # Add seed to all islands
        for island_id in range(cfg.num_islands):
            p = Program(
                id=_make_id(),
                solution=code,
                metrics=dict(result.metrics),
                island_id=island_id,
                iteration=0,
                feedback=seed.feedback,
            )
            self._db.add(p, 0)

        scores = ", ".join(f"{o}={seed.get_objective(o):.4f}" for o in cfg.objectives)
        logger.info("Seed program: %s", scores)

    async def _run_iteration(self, iteration: int) -> None:
        """Run a single iteration: sample -> mutate -> evaluate -> store."""
        cfg = self.config

        for attempt in range(cfg.retry_attempts + 1):
            try:
                # Sample parent and context
                parent, context, explore = self._db.sample(cfg.num_context_programs)

                # Build mutation prompt
                system, user = build_prompt(
                    parent,
                    context,
                    cfg.objectives,
                    cfg.diff_mode,
                    explore,
                )

                # LLM mutation
                response = await generate(
                    model=cfg.model,
                    system=system,
                    user=user,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )

                # Parse response
                child_code = parse_response(
                    response,
                    parent_code=parent.solution if cfg.diff_mode else None,
                    diff_mode=cfg.diff_mode,
                )

                if not child_code:
                    logger.warning(
                        "Iteration %d attempt %d: failed to parse LLM response", iteration, attempt
                    )
                    continue

                # Syntax check
                try:
                    compile(child_code, "<evolved>", "exec")
                except SyntaxError as e:
                    logger.warning(
                        "Iteration %d attempt %d: syntax error: %s", iteration, attempt, e
                    )
                    continue

                # Size cap — reject bloated mutations that break future diffs
                if cfg.max_code_lines > 0:
                    lines = child_code.count("\n") + 1
                    if lines > cfg.max_code_lines:
                        logger.warning(
                            "Iteration %d attempt %d: code too large (%d lines > %d max)",
                            iteration,
                            attempt,
                            lines,
                            cfg.max_code_lines,
                        )
                        continue

                # Evaluate
                child_id = _make_id()
                result = await run_evaluation(self._evaluate_fn, child_code, child_id)

                if result.error:
                    logger.warning(
                        "Iteration %d attempt %d: eval error: %s", iteration, attempt, result.error
                    )
                    continue

                # Store
                child = Program(
                    id=child_id,
                    solution=child_code,
                    metrics=result.metrics,
                    parent_id=parent.id,
                    island_id=parent.island_id,
                    iteration=iteration,
                    feedback=result.artifacts.get("feedback", ""),
                )
                self._db.add(child, iteration)

                scores = ", ".join(f"{o}={child.get_objective(o):.4f}" for o in cfg.objectives)
                logger.info("Iteration %d: child %s (%s)", iteration, child_id[:8], scores)
                return

            except Exception as e:
                logger.error(
                    "Iteration %d attempt %d failed: %s", iteration, attempt, e, exc_info=True
                )

        logger.warning("Iteration %d: all attempts failed", iteration)


def _make_id() -> str:
    return uuid.uuid4().hex[:12]
