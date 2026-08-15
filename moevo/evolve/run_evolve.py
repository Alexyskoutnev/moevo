"""Evolve agent code across GDPval slices.

Supports two evolution engines:
  - skydiscover: scalar fitness (combined_score), single winner per iteration
  - moevo: multi-objective NSGA-II Pareto selection (gdpval + safety as separate objectives)

The loop works like this:
  1. Load the seed agent code (e.g. seeds/openai.py)
  2. For each zipper slice (S1, S2, ...):
     a. Score the current code on a sample of tasks from that slice
     b. Run N iterations of mutation+evaluation
     c. Extract the best code and carry it to the next slice
  3. Save the final evolved code + trajectory of scores

Each slice has completely different tasks, so improvements must generalize -
code that overfits to one slice's tasks will score poorly on the next.

Usage:
    # SkyDiscover (scalar fitness, default)
    python -m moevo.evolve.run_evolve --slices S1 S2 S3 --iterations 5

    # moevo (multi-objective Pareto)
    python -m moevo.evolve.run_evolve --engine moevo --slices S1 S2 S3 --iterations 10

    # moevo with pro mutation model
    python -m moevo.evolve.run_evolve --engine moevo --seed openai --tier slow --mutation-model slow --slices S1 S2 S3 S4 S5 S6 S7 S8
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from moevo.evolve.cli import parse_args, setup_logging

log = logging.getLogger("evolve")


# ── Logging helpers ───────────────────────────────────────────────────────


def _code_stats(code: str) -> str:
    return f"{len(code):,} chars, {code.count(chr(10)) + 1} lines"


def _log_banner(title: str, items: dict[str, str]) -> None:
    sep = "=" * 60
    lines = [sep, title]
    for k, v in items.items():
        lines.append(f"  {k + ':':<18}{v}")
    lines.append(sep)
    log.info("\n".join(lines))


def _log_table(
    header: list[str], rows: list[list[str]], footer: dict[str, str] | None = None
) -> None:
    fmt = "  %-6s %9s %9s %6s"
    lines = ["=" * 60, "DONE", fmt % tuple(header)]
    for row in rows:
        lines.append(fmt % tuple(row))
    if footer:
        for k, v in footer.items():
            lines.append(f"  {k}  {v}")
    lines.append("=" * 60)
    log.info("\n".join(lines))


# ── SkyDiscover engine ───────────────────────────────────────────────────


def _evolve_slice_skydiscover(
    current_code: str,
    slice_name: str,
    model: str,
    search: str,
    iterations: int,
    sample_size: int,
    evaluator_path: str,
    output_dir: Path,
) -> tuple[str, float, float, dict]:
    """Run SkyDiscover on one slice. Returns (best_code, initial_score, best_score, extra)."""
    from skydiscover.api import run_discovery

    from moevo.evolve.config import build_config

    os.environ["EVOLVE_SLICE"] = slice_name

    log.info(
        "Evolving %s | skydiscover (%s) | %s | %d iters, %d tasks",
        slice_name,
        search,
        model,
        iterations,
        sample_size,
    )

    config = build_config(model=model, iterations=iterations, search=search)
    t0 = time.monotonic()

    result = run_discovery(
        evaluator=evaluator_path,
        initial_program=current_code,
        config=config,
        iterations=iterations,
        output_dir=str(output_dir / f"skydiscover_{slice_name}"),
        cleanup=False,
    )

    elapsed = time.monotonic() - t0
    initial = result.initial_score or 0.0
    best = result.best_score
    best_code = result.best_solution or current_code
    improved = best > initial

    log.info(
        "%s done (%.0fs): %.1f%% -> %.1f%% %s",
        slice_name,
        elapsed,
        initial * 100,
        best * 100,
        "(IMPROVED)" if improved else "(no change)",
    )

    return (best_code if improved else current_code), initial, best, {}


# ── moevo engine ─────────────────────────────────────────────────────────


def _evolve_slice_moevo(
    current_code: str,
    slice_name: str,
    model: str,
    search: str,
    iterations: int,
    sample_size: int,
    evaluator_path: str,
    output_dir: Path,
    safety_samples: int = 3,
) -> tuple[str, float, float, dict]:
    """Run moevo on one slice. Returns (best_code, seed_gdpval, best_gdpval, pareto_info)."""
    from moevo import MoevoConfig, MoevoController

    os.environ["EVOLVE_SLICE"] = slice_name

    log.info(
        "Evolving %s | moevo (NSGA-II) | %s | %d iters, %d+%d tasks",
        slice_name,
        model,
        iterations,
        sample_size,
        safety_samples,
    )

    seed_path = output_dir / f"seed_{slice_name}.py"
    seed_path.write_text(current_code)

    config = MoevoConfig(
        initial_program=str(seed_path),
        evaluator_path=evaluator_path,
        objectives=["gdpval_score", "safety_score"],
        model=model,
        iterations=iterations,
        population_size=10,
        num_islands=2,
        migration_interval=5,
        migration_count=2,
        checkpoint_interval=2,
        fresh_start=True,
        max_code_lines=10000,
        output_dir=str(output_dir / f"moevo_{slice_name}"),
    )

    t0 = time.monotonic()
    controller = MoevoController(config)
    result = asyncio.run(controller.run())
    elapsed = time.monotonic() - t0

    front = result.pareto_front

    # Pick the most BALANCED program from the Pareto front to carry forward.
    # Bug fix: previously used best_program (highest HV contribution), which
    # biases toward capability and lets safety degrade across slices.
    # Now we pick the program with the highest geometric mean of objectives,
    # which ensures neither objective is sacrificed.
    best = None
    if front:
        best = max(
            front, key=lambda p: p.get_objective("gdpval_score") * p.get_objective("safety_score")
        )
    elif result.best_program:
        best = result.best_program

    if best:
        best_gdpval = best.get_objective("gdpval_score")
        best_safety = best.get_objective("safety_score")
        best_code = best.solution
    else:
        best_gdpval, best_safety, best_code = 0.0, 0.0, current_code

    seed_gdpval = 0.0
    seed_safety = 0.0
    if result.all_programs:
        seed_gdpval = result.all_programs[0].get_objective("gdpval_score")
        seed_safety = result.all_programs[0].get_objective("safety_score")

    improved = (best_gdpval * best_safety) > (seed_gdpval * seed_safety)

    log.info(
        "%s done (%.0fs): gdpval %.1f%%->%.1f%% | safety %.1f%%->%.1f%% | HV=%.3f | front=%d %s",
        slice_name,
        elapsed,
        seed_gdpval * 100,
        best_gdpval * 100,
        seed_safety * 100,
        best_safety * 100,
        result.hypervolume,
        len(front),
        "(IMPROVED)" if improved else "(no change)",
    )

    pareto_info = {
        "hypervolume": result.hypervolume,
        "front_size": len(front),
        "front": [
            {
                "gdpval": p.get_objective("gdpval_score"),
                "safety": p.get_objective("safety_score"),
                "id": p.id[:8],
            }
            for p in front
        ],
    }

    return (best_code if improved else current_code), seed_gdpval, best_gdpval, pareto_info


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    engine = getattr(args, "engine", "skydiscover")
    output_dir = Path(args.output_dir or f"results/{engine}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(args.verbose, output_dir / "evolve.log")

    # Load seed - either custom file or provider default
    if args.seed_file:
        seed_path = Path(args.seed_file)
    else:
        seed_path = Path(__file__).resolve().parent / "seeds" / f"{args.seed}.py"
    if not seed_path.exists():
        log.error("Seed not found: %s", seed_path)
        sys.exit(1)
    current_code = seed_path.read_text()

    # Configure evaluator env vars
    os.environ["EVOLVE_SAMPLE_SIZE"] = str(args.sample_size)
    os.environ["EVOLVE_WORKING_DIR"] = str(output_dir / "evolve_workspace")
    os.environ["EVOLVE_AGENT_MODEL"] = args.agent_model
    os.environ["EVOLVE_SAFETY_SAMPLES"] = str(args.safety_samples)
    if args.judge_model:
        os.environ["EVOLVE_JUDGE_MODEL"] = args.judge_model

    # Pick evaluator based on engine
    if engine == "moevo":
        evaluator_path = str(Path(__file__).resolve().parent / "moevo_evaluator.py")
        os.environ["EVOLVE_SAFETY_WEIGHT"] = "1.0"  # moevo always evaluates both
        fitness_desc = "gdpval_score + safety_score (NSGA-II Pareto, no weight)"
    else:
        evaluator_path = str(Path(__file__).resolve().parent / "evaluator.py")
        os.environ["EVOLVE_SAFETY_WEIGHT"] = str(args.safety_weight)
        if args.safety_weight > 0:
            fitness_desc = (
                f"{(1 - args.safety_weight):.0%} GDPval + "
                f"{args.safety_weight:.0%} safety (linear scalar)"
            )
        else:
            fitness_desc = "GDPval only (no safety signal)"

    # Validate full coverage: iterations × sample_size ≥ slice_size
    # This ensures round-robin rotation covers all tasks at least once
    # (Gathercole & Ross, 1994; Bartz-Beielstein et al., 2020)
    if args.sample_size > 0:
        min_slice_size = 22  # GDPval tasks per slice
        coverage = args.iterations * args.sample_size
        if coverage < min_slice_size:
            log.warning(
                "WARNING: iterations(%d) x sample_size(%d) = %d < %d tasks. "
                "Not all tasks will be seen during evolution. "
                "Increase --iterations or --sample-size for full coverage.",
                args.iterations,
                args.sample_size,
                coverage,
                min_slice_size,
            )
        else:
            log.info(
                "Coverage check: %d iterations x %d samples = %d >= %d tasks (full coverage by iteration %d)",
                args.iterations,
                args.sample_size,
                coverage,
                min_slice_size,
                (min_slice_size + args.sample_size - 1) // args.sample_size,
            )

    _log_banner(
        f"RSI EVOLVE ({engine.upper()})",
        {
            "Engine": engine,
            "Seed": f"{args.seed} ({_code_stats(current_code)})",
            "Agent model": f"{args.agent_model} ({args.tier} tier)",
            "Mutation model": args.mutation_model,
            "Fitness": fitness_desc,
            "Slices": " -> ".join(args.slices),
            "Config": f"{args.iterations} iters/slice, {args.sample_size} tasks/eval",
            "Output": str(output_dir),
        },
    )

    (output_dir / "harness_initial.py").write_text(current_code)
    trajectory: list[dict] = []

    # Pick the evolve function
    evolve_fn = _evolve_slice_moevo if engine == "moevo" else _evolve_slice_skydiscover

    for i, s in enumerate(args.slices):
        t0 = time.monotonic()
        log.info("── SLICE %s (%d/%d) ──", s, i + 1, len(args.slices))
        (output_dir / f"harness_{s}_input.py").write_text(current_code)

        try:
            current_code, initial, best, extra = evolve_fn(
                current_code,
                s,
                args.mutation_model,
                args.search,
                args.iterations,
                args.sample_size,
                evaluator_path,
                output_dir,
            )
        except Exception:
            log.exception("Evolution failed on %s", s)
            initial, best, extra = 0.0, 0.0, {}

        (output_dir / f"harness_{s}_evolved.py").write_text(current_code)

        # Full-slice re-evaluation: run the evolved harness on ALL tasks in this
        # slice (not just the 3-task search-time sample). This gives a clean,
        # unbiased score for reporting - following EC benchmarking best practice
        # (Bartz-Beielstein et al., 2020).
        full_slice_score = None
        try:
            from moevo.evolve.slices import run_full_slice_eval

            log.info("── FULL RE-EVAL %s (%d tasks) ──", s, 22)
            full_result = run_full_slice_eval(
                code=current_code,
                slice_name=s,
                use_judge=True,
                judge_model=args.judge_model,
                output_dir=output_dir,
                concurrency=3,
            )
            full_slice_score = full_result.get("avg_score")
            log.info("Full re-eval %s: %.1f%% (all tasks)", s, (full_slice_score or 0) * 100)
        except Exception:
            log.exception("Full re-eval failed on %s (non-fatal)", s)

        duration = time.monotonic() - t0
        entry = {
            "slice": s,
            "iteration": i,
            "initial_score": initial,
            "evolved_score": best,
            "full_slice_score": full_slice_score,
            "code_lines": current_code.count("\n") + 1,
            "duration_s": duration,
            "engine": engine,
        }
        if extra:
            entry["pareto"] = extra
        trajectory.append(entry)
        (output_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))

    (output_dir / "harness_final.py").write_text(current_code)

    _log_table(
        header=["Slice", "Search", "FullEval", "Lines"],
        rows=[
            [
                t["slice"],
                f"{t['evolved_score'] * 100:.1f}%",
                f"{t.get('full_slice_score', 0) * 100:.1f}%"
                if t.get("full_slice_score") is not None
                else "-",
                str(t["code_lines"]),
            ]
            for t in trajectory
        ],
        footer={"Output:": str(output_dir), "Harness:": f"{output_dir}/harness_final.py"},
    )


if __name__ == "__main__":
    main()
