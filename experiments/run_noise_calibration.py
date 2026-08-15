#!/usr/bin/env python3
"""Noise calibration: evaluate the same harness on the same slice N times.

Quantifies the variance of the 3-task sampling protocol used during evolution.
This tells us whether a 5-point difference is signal or noise.

Usage:
    python scripts/run_noise_calibration.py results/moevo_balanced_pro/harness_final.py --slice S1 --repeats 10
    python scripts/run_noise_calibration.py results/moevo_balanced_pro/harness_final.py --slice S1 --repeats 20 --tasks-per-eval 3
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from moevo.data.registry import DatasetRegistry
from moevo.eval.runner import GDPvalRunner
from moevo.evolve.evaluator import _load_agent_from_code
from moevo.evolve.slices import load_zipper_slice

logger = logging.getLogger("noise_calibration")


def setup_logging(verbose: bool) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(console)
    for noisy in ("httpx", "httpcore", "openai", "google", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Noise calibration: evaluate same harness N times on sampled tasks",
    )
    parser.add_argument("harness", type=str, help="Path to harness .py file")
    parser.add_argument("--slice", type=str, default="S1", help="Slice to evaluate on")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeated evaluations")
    parser.add_argument(
        "--tasks-per-eval", type=int, default=3, help="Tasks sampled per evaluation (default: 3)"
    )
    parser.add_argument("--concurrency", type=int, default=3, help="Parallel tasks")
    parser.add_argument("--judge-model", type=str, default="gpt-5.4", help="Judge model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for task sampling")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    harness_path = Path(args.harness)
    if not harness_path.exists():
        print(f"Error: {harness_path} not found")
        sys.exit(1)

    setup_logging(args.verbose)
    code = harness_path.read_text()

    # Load agent
    agent = _load_agent_from_code(code)
    if agent is None:
        logger.error("Failed to load agent")
        sys.exit(1)

    # Load slice samples
    registry = DatasetRegistry()
    registry.load_dataset("gdpval")
    all_samples = registry.get_samples("gdpval")
    id_set = set(load_zipper_slice(args.slice))
    slice_samples = [s for s in all_samples if s.id in id_set]

    logger.info("=" * 60)
    logger.info("NOISE CALIBRATION")
    logger.info("  Harness:    %s", harness_path)
    logger.info("  Slice:      %s (%d tasks available)", args.slice, len(slice_samples))
    logger.info("  Repeats:    %d", args.repeats)
    logger.info("  Tasks/eval: %d", args.tasks_per_eval)
    logger.info("  Judge:      %s", args.judge_model)
    logger.info("=" * 60)

    rng = random.Random(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/noise_calibration_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scores = []
    t0 = time.monotonic()

    for i in range(args.repeats):
        # Sample tasks
        sampled = rng.sample(slice_samples, min(args.tasks_per_eval, len(slice_samples)))

        runner = GDPvalRunner(
            agent=agent,
            working_dir=output_dir / f"workspace_rep{i}",
            use_judge=True,
            judge_model=args.judge_model,
        )

        result = asyncio.run(runner.run_batch(sampled, concurrency=args.concurrency))
        score = result.avg_score
        all_scores.append(score)

        logger.info(
            "  Rep %d/%d: %.1f%% (tasks: %s)",
            i + 1,
            args.repeats,
            score * 100,
            [s.id[:8] for s in sampled],
        )

    elapsed = time.monotonic() - t0

    # Compute statistics
    import statistics

    mean = statistics.mean(all_scores)
    stdev = statistics.stdev(all_scores) if len(all_scores) > 1 else 0
    min_score = min(all_scores)
    max_score = max(all_scores)
    scores_pct = [s * 100 for s in all_scores]

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("  Mean:   %.1f%% ± %.1f%%", mean * 100, stdev * 100)
    logger.info("  Range:  %.1f%% - %.1f%%", min_score * 100, max_score * 100)
    logger.info("  Stdev:  %.1f%%", stdev * 100)
    logger.info("  All:    %s", [f"{s:.1f}" for s in scores_pct])
    logger.info("  Time:   %.0fs", elapsed)
    logger.info("=" * 60)

    # Save results
    results = {
        "harness": str(harness_path),
        "slice": args.slice,
        "repeats": args.repeats,
        "tasks_per_eval": args.tasks_per_eval,
        "judge_model": args.judge_model,
        "seed": args.seed,
        "scores": all_scores,
        "mean": mean,
        "stdev": stdev,
        "min": min_score,
        "max": max_score,
        "duration_s": elapsed,
    }
    results_path = output_dir / "noise_calibration.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
