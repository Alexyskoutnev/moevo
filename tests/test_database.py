"""Tests for ParetoDatabase."""

from __future__ import annotations

from moevo.core import Program
from moevo.search import ParetoDatabase


def _make(id: str, island: int = 0, **metrics) -> Program:
    return Program(id=id, solution=f"# {id}", metrics=metrics, island_id=island)


class TestParetoDatabase:
    def test_add_and_retrieve(self):
        db = ParetoDatabase(objectives=["a", "b"], population_size=10, num_islands=2)
        p = _make("p1", island=0, a=0.5, b=0.5)
        db.add(p, iteration=0)
        assert len(db.all_programs) == 1
        assert len(db.islands[0]) == 1

    def test_pruning_keeps_pareto_optimal(self):
        db = ParetoDatabase(objectives=["a", "b"], population_size=4, num_islands=1)
        # Add programs - capacity is 4/1 = 4, min 2
        programs = [
            _make("front1", island=0, a=0.9, b=0.1),
            _make("front2", island=0, a=0.1, b=0.9),
            _make("front3", island=0, a=0.5, b=0.8),
            _make("mid", island=0, a=0.5, b=0.5),
            _make("bad", island=0, a=0.1, b=0.1),
        ]
        for i, p in enumerate(programs):
            db.add(p, iteration=i)

        # Island should be pruned to capacity (4)
        assert len(db.islands[0]) <= 4

        # Pareto-optimal programs should survive
        ids = {p.id for p in db.islands[0]}
        assert "front1" in ids
        assert "front2" in ids

    def test_sample_returns_from_population(self):
        db = ParetoDatabase(objectives=["a"], population_size=10, num_islands=2)
        db.add(_make("p1", island=0, a=0.5), iteration=0)
        db.add(_make("p2", island=1, a=0.8), iteration=0)

        parent, context, explore = db.sample(num_context=1)
        assert parent.id in ("p1", "p2")
        assert isinstance(explore, bool)

    def test_get_pareto_front_global(self):
        db = ParetoDatabase(objectives=["a", "b"], population_size=10, num_islands=2)
        db.add(_make("p1", island=0, a=0.9, b=0.1), iteration=0)
        db.add(_make("p2", island=1, a=0.1, b=0.9), iteration=0)
        db.add(_make("p3", island=0, a=0.1, b=0.1), iteration=0)

        front = db.get_pareto_front()
        ids = {p.id for p in front}
        assert ids == {"p1", "p2"}

    def test_get_best_program(self):
        db = ParetoDatabase(objectives=["a", "b"], population_size=10, num_islands=1)
        db.add(_make("p1", island=0, a=0.9, b=0.9), iteration=0)
        db.add(_make("p2", island=0, a=0.1, b=0.1), iteration=0)

        best = db.get_best_program()
        assert best is not None
        assert best.id == "p1"

    def test_serialization_roundtrip(self):
        db = ParetoDatabase(objectives=["a", "b"], population_size=10, num_islands=2)
        db.add(_make("p1", island=0, a=0.5, b=0.5), iteration=0)
        db.add(_make("p2", island=1, a=0.8, b=0.2), iteration=1)

        data = db.to_dict()
        db2 = ParetoDatabase.from_dict(data)

        assert len(db2.all_programs) == 2
        assert db2.objectives == ["a", "b"]
        assert len(db2.islands[0]) == 1
        assert len(db2.islands[1]) == 1

    def test_migration_happens(self):
        db = ParetoDatabase(
            objectives=["a"],
            population_size=10,
            num_islands=2,
            migration_interval=1,
            migration_count=5,  # Multiple attempts to overcome randomness
        )
        # Populate both islands
        db.add(_make("p1", island=0, a=0.9), iteration=0)
        db.add(_make("p2", island=1, a=0.8), iteration=0)

        initial_0 = len(db.islands[0])
        initial_1 = len(db.islands[1])

        # Trigger migration
        db.end_iteration(1)

        # At least one island should have gained a member
        total = sum(len(isl) for isl in db.islands)
        assert total > initial_0 + initial_1
