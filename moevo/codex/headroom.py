"""Admission rules for evolution objectives, separate from task execution health."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def assess_headroom(
    scores: Sequence[float],
    *,
    infrastructure_errors: int = 0,
    verified_task_failures: int = 0,
    minimum_tasks: int = 32,
    maximum_baseline: float = 0.90,
) -> dict:
    """Require measured task failures, not failed environments, before admission.

    Scores are official binary task successes. Screening uses development data;
    these observations must never be reused as the final held-out result.
    """
    if any(score not in (0, 1) for score in scores):
        raise ValueError("Headroom screening requires binary official task success")
    if infrastructure_errors < 0 or not 0 <= verified_task_failures <= scores.count(0):
        raise ValueError("Invalid failure counts")
    if minimum_tasks < 1 or not 0 < maximum_baseline < 1:
        raise ValueError("Invalid screening thresholds")
    n = len(scores)
    rate = sum(scores) / n if n else None
    interval = None
    if n:
        assert rate is not None
        z = 1.959963984540054
        denominator = 1 + z * z / n
        midpoint = (rate + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
        interval = [max(0, midpoint - radius), min(1, midpoint + radius)]
    if infrastructure_errors:
        status = "repair_infrastructure"
    elif n < minimum_tasks:
        status = "needs_more_screening"
    elif rate == 1:
        status = "saturated_on_screen"
    elif rate is not None and rate > maximum_baseline:
        status = "insufficient_practical_headroom"
    elif verified_task_failures < max(2, math.ceil(n * (1 - maximum_baseline) - 1e-9)):
        status = "audit_task_failures"
    else:
        status = "headroom_confirmed_on_screen"
    return {
        "status": status,
        "tasks_scored": n,
        "baseline_success": rate,
        "wilson_95_interval": interval,
        "infrastructure_errors": infrastructure_errors,
        "verified_task_failures": verified_task_failures,
        "minimum_tasks": minimum_tasks,
        "maximum_baseline": maximum_baseline,
        "final_heldout_evaluation": False,
    }
