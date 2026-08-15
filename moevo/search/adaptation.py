"""UCB island selection and adaptive search intensity with hypervolume reward."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class IslandState:
    """Per-island adaptive state."""

    accumulated_signal: float = 0.0  # G_t: decayed sum of squared deltas
    best_hv: float = 0.0
    raw_visits: int = 0
    decayed_visits: float = 0.0
    decayed_rewards: float = 0.0

    def update(self, hv_delta: float, global_best_hv: float, decay: float) -> None:
        """Update island state after an evaluation on this island."""
        self.raw_visits += 1
        self.decayed_visits = decay * self.decayed_visits + 1.0

        # Local: drives search intensity
        local_delta = self._normalize(hv_delta, self.best_hv)
        self.accumulated_signal = decay * self.accumulated_signal + (1 - decay) * local_delta**2
        if hv_delta > 0:
            self.best_hv += hv_delta

        # Global: drives UCB reward (fair across islands)
        global_delta = self._normalize(hv_delta, global_best_hv)
        self.decayed_rewards = decay * self.decayed_rewards + global_delta

    def receive_migration(self, hv_delta: float, decay: float) -> None:
        """Update state from migration (no visit/reward credit)."""
        local_delta = self._normalize(hv_delta, self.best_hv)
        self.accumulated_signal = decay * self.accumulated_signal + (1 - decay) * local_delta**2
        if hv_delta > 0:
            self.best_hv += hv_delta

    @staticmethod
    def _normalize(delta: float, ref: float, eps: float = 1e-8) -> float:
        if delta <= 0:
            return 0.0
        if ref == 0:
            return 1.0
        return min(delta / (abs(ref) + eps), 1.0)


@dataclass
class AdaptationState:
    """Multi-island adaptation with UCB selection and hypervolume reward."""

    num_islands: int = 2
    ucb_constant: float = 1.41
    decay: float = 0.9
    intensity_min: float = 0.1
    intensity_max: float = 0.7
    global_best_hv: float = 0.0
    total_iterations: int = 0
    islands: list[IslandState] = field(default_factory=list)

    def __post_init__(self):
        if not self.islands:
            self.islands = [IslandState() for _ in range(self.num_islands)]

    def select_island(self) -> int:
        """Select island using UCB1."""
        self.total_iterations += 1

        # Ensure each island is visited at least once
        for i, island in enumerate(self.islands):
            if island.raw_visits == 0:
                return i

        best_idx = 0
        best_score = -math.inf
        ln_total = math.log(self.total_iterations)

        for i, island in enumerate(self.islands):
            avg_reward = island.decayed_rewards / max(island.decayed_visits, 1e-8)
            exploration = self.ucb_constant * math.sqrt(ln_total / island.raw_visits)
            score = avg_reward + exploration
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def get_search_intensity(self, island_id: int) -> float:
        """Get search intensity for an island.

        Low G (stagnating) -> high intensity (explore).
        High G (productive) -> low intensity (exploit).
        """
        G = self.islands[island_id].accumulated_signal
        return self.intensity_min + (self.intensity_max - self.intensity_min) / (
            1 + math.sqrt(G + 1e-8)
        )

    def update(self, island_id: int, hv_delta: float) -> None:
        """Update island state after evaluation."""
        self.islands[island_id].update(hv_delta, self.global_best_hv, self.decay)
        if hv_delta > 0:
            self.global_best_hv += hv_delta

    def migrate(self, source_id: int, dest_id: int, hv_delta: float) -> None:
        """Record a migration event on the destination island."""
        self.islands[dest_id].receive_migration(hv_delta, self.decay)

    def to_dict(self) -> dict:
        return {
            "num_islands": self.num_islands,
            "ucb_constant": self.ucb_constant,
            "decay": self.decay,
            "intensity_min": self.intensity_min,
            "intensity_max": self.intensity_max,
            "global_best_hv": self.global_best_hv,
            "total_iterations": self.total_iterations,
            "islands": [
                {
                    "accumulated_signal": s.accumulated_signal,
                    "best_hv": s.best_hv,
                    "raw_visits": s.raw_visits,
                    "decayed_visits": s.decayed_visits,
                    "decayed_rewards": s.decayed_rewards,
                }
                for s in self.islands
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> AdaptationState:
        state = cls(
            num_islands=d["num_islands"],
            ucb_constant=d["ucb_constant"],
            decay=d["decay"],
            intensity_min=d["intensity_min"],
            intensity_max=d["intensity_max"],
            global_best_hv=d["global_best_hv"],
            total_iterations=d["total_iterations"],
        )
        state.islands = [IslandState(**s) for s in d["islands"]]
        return state
