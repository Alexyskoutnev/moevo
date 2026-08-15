#!/usr/bin/env python3
"""Cross-judge validation: re-score existing traces with a different judge model.

Takes existing traces (from a previous eval run) and re-scores them using
a different judge model to check for same-family bias.

Usage:
    # Re-score moevo pro S1 traces with Claude
    python scripts/run_cross_judge.py results/moevo_balanced_pro/E1/traces.json --judge-model claude-sonnet-4-6

    # Re-score multiple slices
    python scripts/run_cross_judge.py results/moevo_balanced_pro/E1/traces.json results/moevo_balanced_pro/E2/traces.json --judge-model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from moevo.data.registry import DatasetRegistry
from moevo.eval.evaluators.gdpval_judge import GDPvalJudgeEvaluator

logger = logging.getLogger("cross_judge")


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
        description="Re-score existing traces with a different judge model",
    )
    parser.add_argument("traces", nargs="+", help="Path(s) to traces.json files")
    parser.add_argument(
        "--judge-model",
        type=str,
        required=True,
        help="Judge model to use (e.g. claude-sonnet-4-6, gemini-2.5-pro)",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Parallel judge calls")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Load GDPval samples for rubrics
    registry = DatasetRegistry()
    registry.load_dataset("gdpval")
    all_samples = registry.get_samples("gdpval")
    sample_map = {s.id: s for s in all_samples}

    judge = GDPvalJudgeEvaluator(model=args.judge_model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/cross_judge_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_comparisons = []

    for traces_path_str in args.traces:
        traces_path = Path(traces_path_str)
        if not traces_path.exists():
            logger.error("Not found: %s", traces_path)
            continue

        with open(traces_path) as f:
            data = json.load(f)

        traces = data.get("traces", data) if isinstance(data, dict) else data
        slice_name = traces_path.parent.name

        logger.info("=" * 60)
        logger.info("Re-scoring %s (%d traces) with %s", slice_name, len(traces), args.judge_model)

        async def rescore_all(
            _traces=traces,
        ):
            sem = asyncio.Semaphore(args.concurrency)

            async def rescore_one(trace):
                async with sem:
                    task_id = trace["task_id"]
                    response = trace.get("response", "")
                    sample = sample_map.get(task_id)
                    if not sample:
                        logger.warning("  Task %s not found in GDPval samples", task_id[:12])
                        return None

                    try:
                        result = await judge.evaluate(
                            task_id=task_id,
                            prompt=sample.prompt,
                            response=response,
                            reference=sample.reference,
                        )
                        return {
                            "task_id": task_id,
                            "new_score": result.normalized_score,
                            "new_raw": result.score,
                            "max_score": result.max_score,
                        }
                    except Exception as e:
                        logger.error("  Failed %s: %s", task_id[:12], e)
                        return {"task_id": task_id, "new_score": 0.0, "error": str(e)}

            tasks = [rescore_one(t) for t in _traces]
            return [r for r in await asyncio.gather(*tasks) if r is not None]

        t0 = time.monotonic()
        new_scores = asyncio.run(rescore_all())
        elapsed = time.monotonic() - t0

        # Match with original scores
        # Load original eval.json for original scores
        eval_path = traces_path.parent / "eval.json"
        orig_scores_map = {}
        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)
            for s in eval_data.get("scores", []):
                orig_scores_map[s["task_id"]] = s["normalized_score"]

        comparisons = []
        for ns in new_scores:
            tid = ns["task_id"]
            orig = orig_scores_map.get(tid)
            ns["original_score"] = orig
            ns["original_judge"] = "gpt-5.4"
            ns["new_judge"] = args.judge_model
            comparisons.append(ns)

        new_avg = sum(ns["new_score"] for ns in new_scores) / len(new_scores) if new_scores else 0
        orig_avg = (
            sum(ns["original_score"] for ns in new_scores if ns["original_score"] is not None)
            / len(new_scores)
            if new_scores
            else 0
        )

        logger.info("  Original judge (gpt-5.4): %.1f%%", orig_avg * 100)
        logger.info("  New judge (%s): %.1f%%", args.judge_model, new_avg * 100)
        logger.info("  Delta: %+.1f%%", (new_avg - orig_avg) * 100)
        logger.info("  Time: %.0fs", elapsed)

        all_comparisons.append(
            {
                "slice": slice_name,
                "traces_path": str(traces_path),
                "original_judge": "gpt-5.4",
                "new_judge": args.judge_model,
                "original_avg": orig_avg,
                "new_avg": new_avg,
                "delta": new_avg - orig_avg,
                "num_tasks": len(new_scores),
                "per_task": comparisons,
            }
        )

    # Save results
    results_path = output_dir / "cross_judge_results.json"
    with open(results_path, "w") as f:
        json.dump(all_comparisons, f, indent=2)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    for comp in all_comparisons:
        logger.info(
            "  %s: orig=%.1f%% new=%.1f%% delta=%+.1f%%",
            comp["slice"],
            comp["original_avg"] * 100,
            comp["new_avg"] * 100,
            comp["delta"] * 100,
        )
    logger.info("Results saved to %s", results_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
