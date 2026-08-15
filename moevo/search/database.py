"""ParetoDatabase - multi-island population with NSGA-II survival."""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

from ..core.types import Program
from .adaptation import AdaptationState
from .pareto import (
    crowding_distance,
    get_pareto_front,
    hypervolume,
    nsga2_select,
    select_parent,
)

logger = logging.getLogger("moevo.database")


class ParetoDatabase:
    """Multi-island population with Pareto ranking.

    Each island maintains a sub-population. NSGA-II survival prunes islands
    when they exceed capacity. UCB selects which island to evolve on.
    Migration sends Pareto-front members between islands.
    """

    def __init__(
        self,
        objectives: list[str],
        population_size: int = 10,
        num_islands: int = 2,
        migration_interval: int = 10,
        migration_count: int = 1,
        ref_point: list[float] | None = None,
        adaptation: AdaptationState | None = None,
    ):
        self.objectives = objectives
        self.population_size = population_size
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_count = migration_count
        self.ref_point = ref_point

        self.islands: list[list[Program]] = [[] for _ in range(num_islands)]
        self.all_programs: list[Program] = []
        self.adaptation = adaptation or AdaptationState(
            num_islands=num_islands,
        )
        self._rng = np.random.default_rng()
        self._prev_hv = 0.0

    def add(self, program: Program, iteration: int) -> None:
        """Add a program to its assigned island, pruning if over capacity."""
        island_id = program.island_id
        self.islands[island_id].append(program)
        self.all_programs.append(program)

        # Compute HV delta for adaptation
        current_hv = self._global_hypervolume()
        hv_delta = max(0.0, current_hv - self._prev_hv)
        self.adaptation.update(island_id, hv_delta)
        self._prev_hv = current_hv

        # Prune island if over capacity
        cap = self._island_capacity()
        if len(self.islands[island_id]) > cap:
            self.islands[island_id] = nsga2_select(self.islands[island_id], cap, self.objectives)

    def sample(self, num_context: int = 2) -> tuple[Program, list[Program], bool]:
        """Sample a parent and context programs for mutation.

        Returns (parent, context_programs, explore).
        Parent is selected via tournament from the Pareto front of the
        chosen island. Context programs are diverse frontier points.
        explore is True when search intensity says to explore (diversity-driven).
        """
        island_id = self.adaptation.select_island()
        island = self.islands[island_id]

        if not island:
            # Fall back to any non-empty island
            for i, isl in enumerate(self.islands):
                if isl:
                    island = isl
                    island_id = i
                    break
            if not island:
                raise RuntimeError("All islands are empty")

        # Explore vs exploit based on adaptive search intensity
        intensity = self.adaptation.get_search_intensity(island_id)
        explore = random.random() < intensity

        # Select parent from island's Pareto front
        parent = select_parent(island, self.objectives, self._rng)

        # Context: diverse programs from global Pareto front
        global_front = self.get_pareto_front()
        context = self._select_diverse_context(global_front, parent, num_context)

        return parent, context, explore

    def end_iteration(self, iteration: int) -> None:
        """End-of-iteration bookkeeping: migration if due."""
        if (
            self.migration_interval > 0
            and iteration > 0
            and iteration % self.migration_interval == 0
        ):
            self._migrate()

    def get_pareto_front(self) -> list[Program]:
        """Global Pareto front across all islands."""
        all_pop = []
        for island in self.islands:
            all_pop.extend(island)
        return get_pareto_front(all_pop, self.objectives)

    def get_best_program(self) -> Program | None:
        """Program with highest hypervolume contribution (backward compat)."""
        front = self.get_pareto_front()
        if not front:
            return None
        if len(front) == 1:
            return front[0]

        # Find program whose removal decreases HV the most
        full_hv = hypervolume(front, self.objectives, self.ref_point)
        best_contrib = -1.0
        best_prog = front[0]

        for i, prog in enumerate(front):
            subset = front[:i] + front[i + 1 :]
            subset_hv = hypervolume(subset, self.objectives, self.ref_point)
            contrib = full_hv - subset_hv
            if contrib > best_contrib:
                best_contrib = contrib
                best_prog = prog

        return best_prog

    def get_hypervolume(self) -> float:
        """Current global hypervolume."""
        return self._global_hypervolume()

    def _global_hypervolume(self) -> float:
        all_pop = []
        for island in self.islands:
            all_pop.extend(island)
        return hypervolume(all_pop, self.objectives, self.ref_point)

    def _island_capacity(self) -> int:
        """Per-island capacity (total population / num_islands, min 2)."""
        return max(2, self.population_size // self.num_islands)

    def _migrate(self) -> None:
        """Send Pareto-front members between islands."""
        if self.num_islands < 2:
            return

        for _ in range(self.migration_count):
            source_id = random.randrange(self.num_islands)
            dest_id = random.randrange(self.num_islands)
            while dest_id == source_id:
                dest_id = random.randrange(self.num_islands)

            source = self.islands[source_id]
            if not source:
                continue

            front = get_pareto_front(source, self.objectives)
            if not front:
                continue

            migrant = select_parent(front, self.objectives, self._rng)

            migrated = Program(
                id=migrant.id,
                solution=migrant.solution,
                metrics=dict(migrant.metrics),
                parent_id=migrant.parent_id,
                island_id=dest_id,
                iteration=migrant.iteration,
                feedback=migrant.feedback,
                metadata={**migrant.metadata, "migrated_from": source_id},
            )

            hv_before = hypervolume(self.islands[dest_id], self.objectives, self.ref_point)
            self.islands[dest_id].append(migrated)
            hv_after = hypervolume(self.islands[dest_id], self.objectives, self.ref_point)
            hv_delta = max(0.0, hv_after - hv_before)

            self.adaptation.migrate(source_id, dest_id, hv_delta)

            cap = self._island_capacity()
            if len(self.islands[dest_id]) > cap:
                self.islands[dest_id] = nsga2_select(self.islands[dest_id], cap, self.objectives)

            logger.debug(
                "Migrated %s from island %d -> %d (dHV=%.4f)",
                migrant.id[:8],
                source_id,
                dest_id,
                hv_delta,
            )

    def _select_diverse_context(
        self, front: list[Program], parent: Program, n: int
    ) -> list[Program]:
        """Select diverse context programs from the frontier, excluding parent."""
        candidates = [p for p in front if p.id != parent.id]
        if not candidates:
            return []
        if len(candidates) <= n:
            return candidates

        cd = crowding_distance(candidates, self.objectives)
        indices = np.argsort(-cd)[:n]
        return [candidates[i] for i in indices]

    def to_dict(self) -> dict[str, Any]:
        return {
            "objectives": self.objectives,
            "population_size": self.population_size,
            "num_islands": self.num_islands,
            "migration_interval": self.migration_interval,
            "migration_count": self.migration_count,
            "ref_point": self.ref_point,
            "islands": [[p.to_dict() for p in island] for island in self.islands],
            "all_programs": [p.to_dict() for p in self.all_programs],
            "adaptation": self.adaptation.to_dict(),
            "prev_hv": self._prev_hv,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParetoDatabase:
        adaptation = AdaptationState.from_dict(d["adaptation"])
        db = cls(
            objectives=d["objectives"],
            population_size=d["population_size"],
            num_islands=d["num_islands"],
            migration_interval=d["migration_interval"],
            migration_count=d["migration_count"],
            ref_point=d.get("ref_point"),
            adaptation=adaptation,
        )
        db.islands = [[Program.from_dict(p) for p in island] for island in d["islands"]]
        db.all_programs = [Program.from_dict(p) for p in d["all_programs"]]
        db._prev_hv = d.get("prev_hv", 0.0)
        return db
