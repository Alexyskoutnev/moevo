"""Opt-in many-objective survival and single-harness max-min selection.

Uses pymoo's NSGA-III survival; mutation remains MOEvo's code/policy evolution.
This module does not establish that NSGA-III beats NSGA-II on agent benchmarks.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions

if TYPE_CHECKING:
    from ..core.types import Program


@lru_cache(maxsize=16)
def reference_directions(dimensions: int, capacity: int) -> np.ndarray:
    if dimensions < 2 or capacity < dimensions + 1:
        raise ValueError(
            "NSGA-III needs at least two objectives and objective_count + 1 slots per island"
        )
    # Riesz-energy directions support arbitrary population sizes. A fixed design
    # seed makes geometry identical across runs; selection has its own seeded RNG.
    return get_reference_directions("energy", dimensions, capacity, seed=0)


def nsga3_select(
    programs: list[Program], n: int, objectives: list[str], rng: np.random.Generator
) -> list[Program]:
    directions = reference_directions(len(objectives), n)
    if len(programs) <= n:
        return list(programs)
    values = np.array([p.get_objectives(objectives) for p in programs], dtype=float)
    population = Population.new("F", -values, "source_index", np.arange(len(programs)))
    survival = ReferenceDirectionSurvival(directions)
    survivors = survival.do(
        Problem(n_var=1, n_obj=len(objectives)), population, n_survive=n, random_state=rng
    )
    return [programs[int(i)] for i in survivors.get("source_index")]


def balanced_champion(programs: list[Program], objectives: list[str]) -> Program | None:
    """Lexicographic max-min of comparable domain success rates in [0, 1].

    Improve the weakest domain first, then the second weakest, and so on.
    This returns one actual harness, never per-axis maxima across specialists.
    Use common held-back selection tasks; final-test scores must not be inputs.
    """
    if not objectives:
        raise ValueError("At least one domain objective is required")
    if not programs:
        return None

    def key(program: Program) -> tuple[float, ...]:
        scores = program.get_objectives(objectives)
        if any(not 0 <= score <= 1 for score in scores):
            raise ValueError("Balanced selection requires domain success rates in [0, 1]")
        return tuple(sorted(scores))

    return max(programs, key=key)
