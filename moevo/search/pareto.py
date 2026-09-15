"""NSGA-II Pareto ranking and selection via pymoo.

Uses pymoo as a utility library for non-dominated sorting, crowding distance,
and hypervolume - but NOT its optimization loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

if TYPE_CHECKING:
    from ..core.types import Program


def _objectives_matrix(programs: list[Program], objectives: list[str]) -> np.ndarray:
    """Build (N, M) objective matrix. Negated because pymoo minimizes."""
    return np.array(
        [[-p.get_objective(o) for o in objectives] for p in programs],
        dtype=np.float64,
    )


def pareto_rank(programs: list[Program], objectives: list[str]) -> list[list[int]]:
    """Non-dominated sorting into Pareto fronts.

    Returns list of fronts, each a list of indices into `programs`.
    Front 0 is the Pareto-optimal set.
    """
    if not programs:
        return []
    F = _objectives_matrix(programs, objectives)
    nds = NonDominatedSorting()
    fronts = nds.do(F)
    return [np.asarray(front).tolist() for front in fronts]


def crowding_distance(programs: list[Program], objectives: list[str]) -> np.ndarray:
    """Compute crowding distance for each program.

    Returns array of shape (N,) with distance per program. Infinite for
    boundary points. Programs in dominated fronts still get a distance
    relative to their own front.
    """
    if not programs:
        return np.array([])

    F = _objectives_matrix(programs, objectives)
    n, m = F.shape
    dist = np.zeros(n)

    fronts = pareto_rank(programs, objectives)
    for front in fronts:
        if len(front) <= 2:
            dist[front] = np.inf
            continue
        front_F = F[front]
        for j in range(m):
            order = np.argsort(front_F[:, j])
            sorted_idx = [front[i] for i in order]
            f_range = front_F[order[-1], j] - front_F[order[0], j]
            if f_range == 0:
                continue
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            for k in range(1, len(order) - 1):
                dist[sorted_idx[k]] += (
                    front_F[order[k + 1], j] - front_F[order[k - 1], j]
                ) / f_range

    return dist


def nsga2_select(programs: list[Program], n: int, objectives: list[str]) -> list[Program]:
    """NSGA-II survival selection: keep n programs by rank then crowding."""
    if len(programs) <= n:
        return list(programs)

    fronts = pareto_rank(programs, objectives)
    selected: list[int] = []

    for front in fronts:
        if len(selected) + len(front) <= n:
            selected.extend(front)
        else:
            remaining = n - len(selected)
            # Compute crowding for this front only
            front_programs = [programs[i] for i in front]
            cd = crowding_distance(front_programs, objectives)
            # Pick highest crowding distance
            best = np.argsort(-cd)[:remaining]
            selected.extend(front[i] for i in best)
            break

    return [programs[i] for i in selected]


def select_parent(
    programs: list[Program], objectives: list[str], rng: np.random.Generator | None = None
) -> Program:
    """Tournament selection with crowding tiebreak from Pareto front."""
    if len(programs) == 1:
        return programs[0]

    if rng is None:
        rng = np.random.default_rng()
    fronts = pareto_rank(programs, objectives)
    front0_idx = fronts[0] if fronts else list(range(len(programs)))
    front0 = [programs[i] for i in front0_idx]

    if len(front0) == 1:
        return front0[0]

    # Binary tournament on crowding distance within front 0
    cd = crowding_distance(front0, objectives)
    i, j = rng.choice(len(front0), size=2, replace=False)
    return front0[i] if cd[i] >= cd[j] else front0[j]


def hypervolume(
    programs: list[Program], objectives: list[str], ref_point: list[float] | None = None
) -> float:
    """Compute hypervolume indicator for the Pareto front.

    Programs are maximized; ref_point is in maximization space (e.g. [0, 0]).
    Internally negated for pymoo which minimizes.
    """
    if not programs:
        return 0.0

    fronts = pareto_rank(programs, objectives)
    if not fronts:
        return 0.0

    front_idx = fronts[0]
    F = _objectives_matrix(programs, objectives)
    front_F = F[front_idx]

    if ref_point is not None:
        ref = np.array([-r for r in ref_point], dtype=np.float64)
    else:
        ref = np.zeros(len(objectives), dtype=np.float64)

    # 1D case: pymoo's HV can segfault; compute directly
    if len(objectives) == 1:
        return float(max(ref[0] - front_F[:, 0].min(), 0.0))

    try:
        indicator = HV(ref_point=ref)
        value = indicator(front_F)
        return 0.0 if value is None else float(value)
    except Exception:
        return 0.0


def get_pareto_front(programs: list[Program], objectives: list[str]) -> list[Program]:
    """Return only Pareto-optimal (non-dominated) programs."""
    if not programs:
        return []
    fronts = pareto_rank(programs, objectives)
    if not fronts:
        return []
    return [programs[i] for i in fronts[0]]
