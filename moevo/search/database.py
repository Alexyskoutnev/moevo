"""ParetoDatabase - multi-island population with NSGA-II survival."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..core.types import Program
from .adaptation import AdaptationState
from .many_objective import balanced_champion, nsga3_select, reference_directions
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
        random_seed: int | None = None,
        selection: str = "pareto",
        weights: list[float] | None = None,
    ):
        if not objectives or len(set(objectives)) != len(objectives):
            raise ValueError("Objectives must be nonempty and unique")
        if num_islands < 1 or population_size < 2 * num_islands:
            raise ValueError("Each island needs capacity for at least two programs")
        if population_size % num_islands:
            raise ValueError("Population size must be divisible by number of islands")
        if selection not in {"pareto", "scalar", "nsga3"}:
            raise ValueError("Selection must be pareto, nsga3, or scalar")
        if selection == "nsga3":
            reference_directions(len(objectives), population_size // num_islands)
        weights = list(weights) if weights is not None else [1.0] * len(objectives)
        if (
            len(weights) != len(objectives)
            or any(not np.isfinite(w) or w < 0 for w in weights)
            or sum(weights) <= 0
        ):
            raise ValueError("Weights must be finite, nonnegative, and match objectives")
        self.objectives = objectives
        self.selection = selection
        self.weights = [w / sum(weights) for w in weights]
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
        self._rng = np.random.default_rng(random_seed)
        self._prev_hv = 0.0

    def add(self, program: Program, iteration: int) -> None:
        """Add a program to its assigned island, pruning if over capacity."""
        island_id = program.island_id
        program.get_objectives(self.objectives)
        if not 0 <= island_id < self.num_islands:
            raise ValueError("Invalid island ID")
        self.islands[island_id].append(program)
        self.all_programs.append(program)

        # Compute HV delta for adaptation
        current_hv = self._global_hypervolume()
        hv_delta = max(0.0, current_hv - self._prev_hv)
        self.adaptation.update(island_id, hv_delta)
        self._prev_hv = max(self._prev_hv, current_hv)

        # Prune island if over capacity
        cap = self._island_capacity()
        if len(self.islands[island_id]) > cap:
            self.islands[island_id] = self._select(self.islands[island_id], cap)

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
        explore = bool(self._rng.random() < intensity)

        # Select parent from island's Pareto front
        parent = (
            select_parent(island, self.objectives, self._rng)
            if self.selection == "pareto"
            else max(island, key=self._scalar_score)
        )
        if self.selection == "nsga3":
            frontier = get_pareto_front(island, self.objectives)
            parent = frontier[int(self._rng.integers(len(frontier)))]

        # Context: diverse programs from global Pareto front
        global_front = (
            self.get_pareto_front()
            if self.selection in {"pareto", "nsga3"}
            else [p for isl in self.islands for p in isl]
        )
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
        unique = list({p.id: p for p in all_pop}.values())
        return get_pareto_front(unique, self.objectives)

    def get_balanced_program(self) -> Program | None:
        """One champion selected by its weakest domain, for normalized success rates."""
        return balanced_champion(self.get_pareto_front(), self.objectives)

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

    def _scalar_score(self, program: Program) -> float:
        return sum(
            w * program.get_objective(o) for w, o in zip(self.weights, self.objectives, strict=True)
        )

    def _select(self, programs: list[Program], n: int) -> list[Program]:
        if self.selection == "nsga3":
            return nsga3_select(programs, n, self.objectives, self._rng)
        if self.selection == "scalar":
            return sorted(programs, key=self._scalar_score, reverse=True)[:n]
        return nsga2_select(programs, n, self.objectives)

    def _migrate(self) -> None:
        """Send Pareto-front members between islands."""
        if self.num_islands < 2:
            return

        for _ in range(self.migration_count):
            source_id = int(self._rng.integers(self.num_islands))
            dest_id = int(self._rng.integers(self.num_islands))
            while dest_id == source_id:
                dest_id = int(self._rng.integers(self.num_islands))

            source = self.islands[source_id]
            if not source:
                continue

            front = get_pareto_front(source, self.objectives)
            if not front:
                continue

            if self.selection == "scalar":
                migrant = max(source, key=self._scalar_score)
            elif self.selection == "nsga3":
                migrant = front[int(self._rng.integers(len(front)))]
            else:
                migrant = select_parent(front, self.objectives, self._rng)
            if any(p.id == migrant.id for p in self.islands[dest_id]):
                continue

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
                self.islands[dest_id] = self._select(self.islands[dest_id], cap)

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
        candidates = list({p.id: p for p in front if p.id != parent.id}.values())
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
            "rng_state": self._rng.bit_generator.state,
            "selection": self.selection,
            "weights": self.weights,
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
            selection=d.get("selection", "pareto"),
            weights=d.get("weights"),
        )
        db.islands = [[Program.from_dict(p) for p in island] for island in d["islands"]]
        db.all_programs = [Program.from_dict(p) for p in d["all_programs"]]
        db._prev_hv = d.get("prev_hv", 0.0)
        if "rng_state" in d:
            db._rng.bit_generator.state = d["rng_state"]
        return db
