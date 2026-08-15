#!/usr/bin/env python3
"""Run full eval on specific slices using an evolved harness.

Usage:
    python scripts/run_eval_slices.py results/evolve_20260315_171007/harness_final.py E1 E2
    python scripts/run_eval_slices.py results/evolve_20260315_150403/harness_final.py S7 S8 E1 E2
    python scripts/run_eval_slices.py harness.py E1 --concurrency 5 --judge-model gpt-5.4
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from moevo.evolve.slices import run_full_slice_eval


def setup_logging(harness_path: Path, verbose: bool) -> None:
    """Configure logging to console and a log file next to the harness."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    log_path = harness_path.parent / "eval_slices.log"
    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)

    for noisy in ("httpx", "httpcore", "openai", "google", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("evolve").info("Logging to %s", log_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run full eval on specific slices using an evolved harness",
    )
    parser.add_argument("harness", type=str, help="Path to harness .py file")
    parser.add_argument("slices", nargs="+", help="Slices to evaluate (e.g. E1 E2)")
    parser.add_argument(
        "--concurrency", type=int, default=3, help="Parallel tasks per slice (default: 3)"
    )
    parser.add_argument(
        "--judge-model", type=str, default="gpt-5.4", help="Judge model (default: gpt-5.4)"
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="Use keyword heuristic instead of LLM judge"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    harness = Path(args.harness)
    if not harness.exists():
        print(f"Error: {harness} not found")
        sys.exit(1)

    code = harness.read_text()
    output = harness.parent

    setup_logging(harness, args.verbose)
    log = logging.getLogger("evolve")

    evaluator = "keyword heuristic" if args.no_judge else f"LLM judge ({args.judge_model})"
    log.info("=" * 60)
    log.info("EVAL SLICES")
    log.info("  Harness:     %s (%d lines)", harness, code.count("\n") + 1)
    log.info("  Slices:      %s", ", ".join(args.slices))
    log.info("  Concurrency: %d", args.concurrency)
    log.info("  Evaluator:   %s", evaluator)
    log.info("  Output:      %s", output)
    log.info("=" * 60)

    total_start = time.monotonic()
    all_results = {}

    for s in args.slices:
        log.info("━━━ Starting %s (22 tasks) ━━━", s)
        result = run_full_slice_eval(
            code,
            s,
            use_judge=not args.no_judge,
            judge_model=args.judge_model,
            output_dir=output,
            concurrency=args.concurrency,
        )
        all_results[s] = result
        log.info(
            "━━━ %s: %.1f%% (%d/%d completed, %d errors, %.0fs) ━━━",
            s,
            result["avg_score"] * 100,
            result["num_completed"],
            result["num_tasks"],
            result["num_errors"],
            result.get("duration_s", 0),
        )
        log.info("")

    total_elapsed = time.monotonic() - total_start

    # Summary table
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    scores = []
    for s, r in all_results.items():
        log.info(
            "  %s: %.1f%% (%d/%d completed, %d errors)",
            s,
            r["avg_score"] * 100,
            r["num_completed"],
            r["num_tasks"],
            r["num_errors"],
        )
        scores.append(r["avg_score"])
    if scores:
        avg = sum(scores) / len(scores)
        log.info("  Avg: %.1f%%", avg * 100)
    log.info("  Total time: %.0fs", total_elapsed)
    log.info("  Harness: %s", harness)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
