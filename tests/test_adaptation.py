"""Tests for UCB island selection and adaptive intensity."""

from __future__ import annotations

from moevo.search import AdaptationState, IslandState


class TestIslandState:
    def test_initial_state(self):
        s = IslandState()
        assert s.accumulated_signal == 0.0
        assert s.raw_visits == 0

    def test_update_with_improvement(self):
        s = IslandState()
        s.update(hv_delta=0.1, global_best_hv=0.5, decay=0.9)
        assert s.raw_visits == 1
        assert s.best_hv == 0.1
        assert s.decayed_rewards > 0

    def test_update_no_improvement(self):
        s = IslandState()
        s.update(hv_delta=0.0, global_best_hv=0.5, decay=0.9)
        assert s.raw_visits == 1
        assert s.best_hv == 0.0
        assert s.decayed_rewards == 0.0

    def test_migration_no_visit_credit(self):
        s = IslandState()
        s.receive_migration(hv_delta=0.1, decay=0.9)
        assert s.raw_visits == 0  # No visit credit
        assert s.best_hv == 0.1


class TestAdaptationState:
    def test_round_robin_initially(self):
        state = AdaptationState(num_islands=3)
        # Each select + update cycle should visit a new island
        visited = set()
        for _ in range(3):
            idx = state.select_island()
            visited.add(idx)
            state.update(idx, hv_delta=0.0)  # Mark as visited
        assert visited == {0, 1, 2}

    def test_ucb_prefers_rewarded_island(self):
        state = AdaptationState(num_islands=2, ucb_constant=0.1)
        # Visit both islands initially
        idx0 = state.select_island()
        state.update(idx0, hv_delta=0.0)
        idx1 = state.select_island()
        state.update(idx1, hv_delta=0.0)
        # Reward island 0 heavily
        for _ in range(10):
            state.update(island_id=0, hv_delta=0.5)

        # Island 0 should be preferred
        selections = []
        for _ in range(20):
            idx = state.select_island()
            selections.append(idx)
            state.update(idx, hv_delta=0.0)
        assert selections.count(0) > selections.count(1)

    def test_search_intensity_high_when_stagnating(self):
        state = AdaptationState(num_islands=1, intensity_min=0.1, intensity_max=0.7)
        # No updates → G ≈ 0 → high intensity
        intensity = state.get_search_intensity(0)
        assert intensity > 0.5

    def test_search_intensity_decreases_with_productivity(self):
        state = AdaptationState(num_islands=1, intensity_min=0.1, intensity_max=0.7)
        initial_intensity = state.get_search_intensity(0)
        # Many improvements → G increases → intensity should decrease
        for _ in range(50):
            state.update(0, hv_delta=1.0)
        intensity = state.get_search_intensity(0)
        assert intensity < initial_intensity

    def test_serialization_roundtrip(self):
        state = AdaptationState(num_islands=2)
        state.select_island()
        state.update(0, hv_delta=0.5)

        data = state.to_dict()
        restored = AdaptationState.from_dict(data)

        assert restored.num_islands == 2
        assert restored.global_best_hv == state.global_best_hv
        assert restored.islands[0].raw_visits == state.islands[0].raw_visits
