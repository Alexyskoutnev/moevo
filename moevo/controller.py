"""Main evolution loop for moevo."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from .core import DiscoveryResult, MoevoConfig, Program
from .core.checkpoint import (
    checkpoint_evaluation_signature,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from .generation import build_prompt, generate, load_evaluate_fn, parse_response, run_evaluation
from .generation.schedule import EvaluationBudgetError, StagedEvaluator
from .search import ParetoDatabase
from .search.adaptation import AdaptationState

logger = logging.getLogger("moevo.controller")


class MoevoController:
    """Orchestrates the evolution loop."""

    def __init__(self, config: MoevoConfig):
        self.config = config
        self._evaluate_fn = None
        self._db: ParetoDatabase | None = None
        self._schedule: StagedEvaluator | None = None

    async def run(self) -> DiscoveryResult:
        """Run the full evolution loop."""
        cfg = self.config
        if cfg.iterations < 0 or cfg.checkpoint_interval < 1:
            raise ValueError("Iterations must be nonnegative and checkpoint interval positive")
        self._db = None
        self._schedule = None
        self._evaluate_fn = None
        output_dir = cfg.output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load evaluator
        if cfg.evaluation_manifest:
            task_fn = load_evaluate_fn(cfg.evaluator_path, "evaluate_task")
            self._schedule = StagedEvaluator(
                task_fn,
                manifest_path=cfg.evaluation_manifest,
                evaluator_path=cfg.evaluator_path,
                objectives=cfg.objectives,
                model=cfg.model,
                judge_model=cfg.judge_model,
                judge_reasoning_effort=cfg.judge_reasoning_effort,
                concurrency=cfg.evaluation_concurrency,
                output_dir=output_dir,
                screen_domains=cfg.screen_domains,
                screen_tasks=cfg.screen_tasks_per_domain,
                audit_every=cfg.screen_audit_every,
                max_task_evaluations=cfg.max_task_evaluations,
                random_seed=cfg.random_seed,
                fresh_start=cfg.fresh_start,
            )
        else:
            self._evaluate_fn = load_evaluate_fn(cfg.evaluator_path)

        # Initialize fresh (don't resume from stale checkpoints in cascade mode -
        # each slice should start clean with its own seed evaluation)
        start_iteration = 0
        if not cfg.fresh_start:
            latest = find_latest_checkpoint(output_dir)
            if latest:
                signature = self._schedule.signature if self._schedule else None
                if checkpoint_evaluation_signature(latest) != signature:
                    raise ValueError("Checkpoint uses a different evaluation protocol")
                self._db, start_iteration = load_checkpoint(latest)
                if self._db.objectives != cfg.objectives or self._db.selection != cfg.selection:
                    raise ValueError(
                        "Checkpoint objectives/selection differ from the run configuration"
                    )
                start_iteration += 1
                logger.info("Resuming from iteration %d", start_iteration)
        if self._db is None:
            self._db = ParetoDatabase(
                objectives=cfg.objectives,
                population_size=cfg.population_size,
                num_islands=cfg.num_islands,
                migration_interval=cfg.migration_interval,
                migration_count=cfg.migration_count,
                ref_point=cfg.ref_point,
                random_seed=cfg.random_seed,
                selection=cfg.selection,
                weights=cfg.weights,
                adaptation=AdaptationState(
                    num_islands=cfg.num_islands,
                    ucb_constant=cfg.ucb_constant,
                    decay=cfg.decay,
                    intensity_min=cfg.intensity_min,
                    intensity_max=cfg.intensity_max,
                ),
            )
            # Seed with initial program
            await self._seed_initial_program()
            self._checkpoint(-1)

        assert self._db is not None
        # Main loop
        iterations_completed = start_iteration
        stop_reason = "iterations_completed"
        for iteration in range(start_iteration, cfg.iterations):
            logger.info("=== Iteration %d/%d ===", iteration + 1, cfg.iterations)
            try:
                await self._run_iteration(iteration)
            except EvaluationBudgetError as exc:
                logger.info("Stopping at the task budget: %s", exc)
                stop_reason = "task_evaluation_budget"
                break

            iterations_completed = iteration + 1

            self._db.end_iteration(iteration + 1)

            # Log status
            front = self._db.get_pareto_front()
            hv = self._db.get_hypervolume()
            logger.info("Pareto front: %d programs, HV=%.4f", len(front), hv)
            for p in front:
                scores = ", ".join(f"{o}={p.get_objective(o):.4f}" for o in cfg.objectives)
                logger.info("  %s: %s", p.id[:8], scores)

            # Checkpoint
            if self._schedule or (iteration + 1) % cfg.checkpoint_interval == 0:
                self._checkpoint(iteration)

        # Final checkpoint
        self._checkpoint(iterations_completed - 1)

        # Save best program
        best = self._db.get_balanced_program() if self._schedule else self._db.get_best_program()
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
            iterations_completed=iterations_completed,
            hypervolume=self._db.get_hypervolume(),
            stop_reason=stop_reason,
            task_evaluations=self._schedule.task_evaluations if self._schedule else None,
        )

    def _checkpoint(self, iteration: int) -> None:
        assert self._db is not None
        save_checkpoint(
            self._db,
            iteration,
            self.config.output_path,
            evaluation_signature=self._schedule.signature if self._schedule else None,
        )

    async def _seed_initial_program(self) -> None:
        """Evaluate and add the initial seed program."""
        assert self._db is not None
        cfg = self.config
        code = Path(cfg.initial_program).read_text()
        program_id = _make_id()

        logger.info("Evaluating seed program...")
        result = (
            await self._schedule.confirm(code)
            if self._schedule
            else await run_evaluation(self._evaluate_fn, code, program_id)
        )

        if result.error:
            raise RuntimeError(f"Seed evaluation failed: {result.error}")

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
                metadata={"evaluation_signature": result.artifacts.get("evaluation_signature")},
            )
            self._db.add(p, 0)

        scores = ", ".join(f"{o}={seed.get_objective(o):.4f}" for o in cfg.objectives)
        logger.info("Seed program: %s", scores)

    async def _run_iteration(self, iteration: int) -> None:
        """Run a single iteration: sample -> mutate -> evaluate -> store."""
        assert self._db is not None
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
                if self._schedule:
                    domains = dict.fromkeys(
                        self._schedule.tasks[t]["objective"]
                        for t in self._schedule.screen_ids(self._schedule.state["screens"])
                    )
                    system += (
                        "\nEvolve one shared harness for every listed domain. Keep the model, "
                        "account authentication, per-task resource limits, benchmark inputs, "
                        "and graders fixed. Improve reusable behavior rather than hardcoding "
                        "task answers or benchmark IDs."
                    )
                    if self._schedule.mutation_scope:
                        system += "\n" + self._schedule.mutation_scope
                    user += (
                        f"\nNext paired search screen: {', '.join(domains)}. "
                        "Broader evaluation covers every domain before population admission."
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

                if any(p.solution == child_code for p in self._db.all_programs) or (
                    self._schedule and self._schedule.seen(child_code)
                ):
                    logger.info("Skipping duplicate candidate before evaluation")
                    continue

                # Syntax check
                try:
                    compile(child_code, "<evolved>", "exec")
                except SyntaxError as e:
                    logger.warning(
                        "Iteration %d attempt %d: syntax error: %s", iteration, attempt, e
                    )
                    continue

                # Size cap - reject bloated mutations that break future diffs
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
                if self._schedule:
                    result = await self._schedule.candidate(
                        child_code, parent.solution, parent.metrics
                    )
                    if result is None:
                        logger.info("Candidate did not pass the paired screen")
                        return
                else:
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
                    metadata={"evaluation_signature": result.artifacts.get("evaluation_signature")},
                )
                self._db.add(child, iteration)

                scores = ", ".join(f"{o}={child.get_objective(o):.4f}" for o in cfg.objectives)
                logger.info("Iteration %d: child %s (%s)", iteration, child_id[:8], scores)
                return

            except EvaluationBudgetError:
                raise
            except Exception as e:
                logger.error(
                    "Iteration %d attempt %d failed: %s", iteration, attempt, e, exc_info=True
                )

        logger.warning("Iteration %d: all attempts failed", iteration)


def _make_id() -> str:
    return uuid.uuid4().hex[:12]
