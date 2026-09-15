"""Aggregate a frozen task panel without mixing partial candidates or cache repeats."""

from __future__ import annotations

import math
from collections import defaultdict

METRICS = {
    "finqa": "Answer accuracy",
    "bizfinbench2": "Numeric answer accuracy",
    "genebench_pro": "Composite task score",
    "amo": "Answer accuracy",
    "putnambench": "Verified proof rate",
    "travelplanner": "All-constraint pass rate",
    "oragentbench": "Feasibility and solution-quality reward",
    "dsbench": "Answer accuracy · Terra judge",
    "gdpval": "Rubric score · Terra judge",
    "terminal_bench_2": "Task pass rate",
    "harvey_lab": "Rubric criteria met · Terra judge",
    "healthbench_professional": "Clipped rubric score · Terra judge",
    "tau3_bench": "Native task reward",
}


def _score(row: dict, native: bool = False) -> float:
    value = row.get("native_metrics", {}).get("score") if native else row["score"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Report scores must be finite numbers")
    if not native and not 0 <= value <= 1:
        raise ValueError("Normalized report scores must be in [0, 1]")
    return float(value)


def aggregate_panel(
    tasks: list[dict], cache: dict, baseline_hash: str | None, candidate_hash: str | None
) -> dict:
    """Return complete-panel means and explicit task/evaluation denominators.

    Scores match the running optimizer: arithmetic task-evaluation means within
    each domain, then equal weight per domain. A repeated seed is an evaluation,
    not an additional independent task. Native metrics remain separate.
    """
    if len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate panel evaluation IDs")
    observations = {}
    for row in cache.values():
        key = (row.get("candidate_sha256"), row["task_id"])
        if key in observations:
            raise ValueError("Duplicate candidate/task observation in cache")
        observations[key] = row

    def summarize(group: list[dict]) -> dict:
        base = [observations.get((baseline_hash, t["id"])) for t in group]
        current = [observations.get((candidate_hash, t["id"])) for t in group]

        def average(rows, native=False):
            if not rows or any(r is None for r in rows):
                return None
            if native and any(r.get("native_metrics", {}).get("score") is None for r in rows):
                return None
            return sum(_score(r, native) for r in rows) / len(rows)

        start, end = average(base), average(current)
        return {
            "n_tasks": len({(t["benchmark"], t["task_id"]) for t in group}),
            "n_evaluations": len(group),
            "baseline_evaluated": sum(r is not None for r in base),
            "candidate_evaluated": sum(r is not None for r in current),
            "baseline": start,
            "candidate": end,
            "delta_pp": 100 * (end - start) if start is not None and end is not None else None,
            "baseline_native": average(base, native=True),
            "candidate_native": average(current, native=True),
        }

    benchmark_tasks, domain_tasks = defaultdict(list), defaultdict(list)
    for task in tasks:
        benchmark_tasks[(task["objective"], task["benchmark"])].append(task)
        domain_tasks[task["objective"]].append(task)
    benchmarks = [
        {
            "benchmark": benchmark,
            "domain": domain,
            "metric": METRICS.get(benchmark, "Normalized task score"),
            **summarize(group),
        }
        for (domain, benchmark), group in benchmark_tasks.items()
    ]
    domains = [{"domain": domain, **summarize(group)} for domain, group in domain_tasks.items()]

    def macro(field):
        if not domains or any(d[field] is None for d in domains):
            return None
        return sum(d[field] for d in domains) / len(domains)

    start, end = macro("baseline"), macro("candidate")
    return {
        "benchmarks": benchmarks,
        "domains": domains,
        "overall": {
            "baseline": start,
            "candidate": end,
            "delta_pp": 100 * (end - start) if start is not None and end is not None else None,
            "n_tasks": len({(t["benchmark"], t["task_id"]) for t in tasks}),
            "n_evaluations": len(tasks),
            "n_domains": len(domains),
            "weighting": "Equal domain weights; arithmetic mean of fixed task evaluations within each domain",
        },
    }
