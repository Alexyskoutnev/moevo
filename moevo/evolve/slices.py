from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger("evolve")


def load_zipper_slice(slice_name: str) -> list[str]:
    with open("data/processed/zipper_split.json") as f:
        splits = json.load(f)
    all_slices = {**splits.get("dev_slices", {}), **splits.get("eval_slices", {})}
    if slice_name not in all_slices:
        raise KeyError(f"Slice '{slice_name}' not found. Available: {list(all_slices.keys())}")
    return all_slices[slice_name]


def run_full_slice_eval(
    code: str,
    slice_name: str,
    use_judge: bool,
    judge_model: str | None,
    output_dir: Path,
    concurrency: int = 3,
) -> dict:
    """Run a full 22-task evaluation with the evolved agent code."""
    from moevo.data.registry import DatasetRegistry
    from moevo.eval.runner import GDPvalRunner
    from moevo.evolve.evaluator import _load_agent_from_code

    registry = DatasetRegistry()
    registry.load_dataset("gdpval")
    all_samples = registry.get_samples("gdpval")
    id_set = set(load_zipper_slice(slice_name))
    samples = [s for s in all_samples if s.id in id_set]

    agent = _load_agent_from_code(code)
    if agent is None:
        log.error("[%s] Failed to load agent from code", slice_name)
        return {"slice": slice_name, "avg_score": 0.0, "error": "load failed"}

    log.info("[%s] Agent loaded: %s", slice_name, agent.name())
    log.info(
        "[%s] Starting %d tasks (concurrency=%d, judge=%s)",
        slice_name,
        len(samples),
        concurrency,
        judge_model,
    )

    runner = GDPvalRunner(
        agent=agent,
        working_dir=output_dir / f"workspace_{slice_name}",
        use_judge=use_judge,
        judge_model=judge_model,
    )

    slice_start = time.monotonic()
    scores_so_far: list[float] = []

    def on_progress(done: int, total: int, trace) -> None:
        elapsed = time.monotonic() - slice_start
        rate = done / (elapsed / 60) if elapsed > 0 else 0
        eta_min = (total - done) / rate if rate > 0 else 0

        score_str = ""
        err_str = ""
        if trace and trace.eval_result and trace.eval_result.max_score > 0:
            s = trace.eval_result.normalized_score
            scores_so_far.append(s)
            score_str = f" | score: {s:.0%}"
        elif trace and trace.error:
            scores_so_far.append(0.0)
            err_str = " (ERROR)"

        running_avg = sum(scores_so_far) / len(scores_so_far) if scores_so_far else 0
        log.info(
            "[%s] %d/%d done (%.1f tasks/min, ETA %.0fm, avg %.1f%%)%s%s",
            slice_name,
            done,
            total,
            rate,
            eta_min,
            running_avg * 100,
            score_str,
            err_str,
        )

    result = asyncio.run(
        runner.run_batch(
            samples,
            concurrency=concurrency,
            progress_callback=on_progress,
        )
    )
    GDPvalRunner.save_results(result, output_dir / slice_name)

    elapsed = time.monotonic() - slice_start
    log.info(
        "[%s] DONE — %.1f%% (%d/%d completed, %d errors, %.0fs)",
        slice_name,
        result.avg_score * 100,
        result.num_completed,
        len(result.traces),
        result.num_errors,
        elapsed,
    )

    return {
        "slice": slice_name,
        "avg_score": result.avg_score,
        "num_tasks": len(result.traces),
        "num_completed": result.num_completed,
        "num_errors": result.num_errors,
        "duration_s": result.total_duration_s,
    }
