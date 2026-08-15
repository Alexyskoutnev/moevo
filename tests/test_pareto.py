"""Tests for Pareto ranking, crowding distance, and NSGA-II selection."""

from __future__ import annotations

import numpy as np

from moevo.core import Program
from moevo.search import (
    crowding_distance,
    get_pareto_front,
    hypervolume,
    nsga2_select,
    pareto_rank,
    select_parent,
)


def _make(id: str, **metrics) -> Program:
    return Program(id=id, solution="", metrics=metrics)


class TestParetoRank:
    def test_empty(self):
        assert pareto_rank([], ["a", "b"]) == []

    def test_single(self):
        fronts = pareto_rank([_make("p1", a=1.0, b=1.0)], ["a", "b"])
        assert len(fronts) == 1
        assert fronts[0] == [0]

    def test_two_objectives_clear_dominance(self):
        programs = [
            _make("dominated", a=0.2, b=0.2),
            _make("dominant", a=0.8, b=0.8),
        ]
        fronts = pareto_rank(programs, ["a", "b"])
        assert len(fronts) == 2
        assert fronts[0] == [1]  # dominant is front 0
        assert fronts[1] == [0]  # dominated is front 1

    def test_pareto_front_tradeoff(self):
        """Two programs that trade off — both on front 0."""
        programs = [
            _make("high_a", a=0.9, b=0.1),
            _make("high_b", a=0.1, b=0.9),
        ]
        fronts = pareto_rank(programs, ["a", "b"])
        assert len(fronts) == 1
        assert set(fronts[0]) == {0, 1}

    def test_20_random_programs(self):
        """Verify manual Pareto front matches NSGA-II ranking."""
        rng = np.random.default_rng(42)
        programs = [
            _make(f"p{i}", score1=float(rng.random()), score2=float(rng.random()))
            for i in range(20)
        ]
        fronts = pareto_rank(programs, ["score1", "score2"])

        # Manual check: front 0 should be non-dominated
        front0 = [programs[i] for i in fronts[0]]
        for p in front0:
            for q in programs:
                if q.id == p.id:
                    continue
                # q should NOT dominate p
                assert not (
                    q.get_objective("score1") >= p.get_objective("score1")
                    and q.get_objective("score2") >= p.get_objective("score2")
                    and (
                        q.get_objective("score1") > p.get_objective("score1")
                        or q.get_objective("score2") > p.get_objective("score2")
                    )
                ), f"{q.id} dominates {p.id} but {p.id} is in front 0"


class TestCrowdingDistance:
    def test_boundary_points_infinite(self):
        programs = [
            _make("p1", a=0.0, b=1.0),
            _make("p2", a=0.5, b=0.5),
            _make("p3", a=1.0, b=0.0),
        ]
        cd = crowding_distance(programs, ["a", "b"])
        assert np.isinf(cd[0])
        assert np.isinf(cd[2])
        assert np.isfinite(cd[1])

    def test_two_programs_both_infinite(self):
        programs = [_make("p1", a=0.0, b=1.0), _make("p2", a=1.0, b=0.0)]
        cd = crowding_distance(programs, ["a", "b"])
        assert np.isinf(cd[0])
        assert np.isinf(cd[1])


class TestNSGA2Select:
    def test_keeps_pareto_optimal(self):
        programs = [
            _make("front", a=0.9, b=0.9),
            _make("ok", a=0.5, b=0.5),
            _make("bad", a=0.1, b=0.1),
        ]
        selected = nsga2_select(programs, 2, ["a", "b"])
        ids = {p.id for p in selected}
        assert "front" in ids
        assert len(selected) == 2

    def test_no_pruning_needed(self):
        programs = [_make("p1", a=0.5, b=0.5)]
        selected = nsga2_select(programs, 5, ["a", "b"])
        assert len(selected) == 1


class TestSelectParent:
    def test_single_program(self):
        p = _make("only", a=1.0)
        assert select_parent([p], ["a"]) is p

    def test_returns_from_front(self):
        programs = [
            _make("front1", a=0.9, b=0.1),
            _make("front2", a=0.1, b=0.9),
            _make("dominated", a=0.1, b=0.1),
        ]
        rng = np.random.default_rng(42)
        parent = select_parent(programs, ["a", "b"], rng)
        assert parent.id in ("front1", "front2")


class TestHypervolume:
    def test_empty(self):
        assert hypervolume([], ["a", "b"]) == 0.0

    def test_single_point(self):
        programs = [_make("p1", a=0.5, b=0.5)]
        hv = hypervolume(programs, ["a", "b"], ref_point=[0.0, 0.0])
        assert hv > 0

    def test_dominated_point_excluded(self):
        """HV should be same with or without dominated point."""
        front = [_make("p1", a=0.8, b=0.8)]
        with_dom = front + [_make("p2", a=0.2, b=0.2)]
        hv1 = hypervolume(front, ["a", "b"], ref_point=[0.0, 0.0])
        hv2 = hypervolume(with_dom, ["a", "b"], ref_point=[0.0, 0.0])
        assert abs(hv1 - hv2) < 1e-10


class TestGetParetoFront:
    def test_returns_non_dominated(self):
        programs = [
            _make("a", x=0.9, y=0.1),
            _make("b", x=0.1, y=0.9),
            _make("c", x=0.1, y=0.1),
        ]
        front = get_pareto_front(programs, ["x", "y"])
        ids = {p.id for p in front}
        assert ids == {"a", "b"}
