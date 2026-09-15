import numpy as np
import pytest

from moevo.core.types import Program
from moevo.search.database import ParetoDatabase
from moevo.search.many_objective import balanced_champion, nsga3_select


def program(name, scores):
    return Program(id=name, solution=name, metrics={f"d{i}": s for i, s in enumerate(scores)})


def test_super_harness_prefers_weakest_domain_over_average():
    specialist = program("specialist", [1, 1, 1, 1, 1, 1, 0.1])
    generalist = program("generalist", [0.8] * 7)
    assert balanced_champion([specialist, generalist], list(generalist.metrics)) is generalist
    with pytest.raises(ValueError):
        balanced_champion([program("unscaled", [50, 80])], ["d0", "d1"])


def test_nsga3_survival_excludes_dominated_candidate_and_reproduces():
    objectives = [f"d{i}" for i in range(7)]
    programs = [program(str(i), [0.1 + 0.1 * i] * 7) for i in range(10)]
    first = nsga3_select(programs, 8, objectives, np.random.default_rng(7))
    second = nsga3_select(programs, 8, objectives, np.random.default_rng(7))
    assert len(first) == 8
    assert [p.id for p in first] == [p.id for p in second]
    assert {p.id for p in first} == {str(i) for i in range(2, 10)}


def test_nsga3_checkpoint_preserves_next_selection():
    objectives = [f"d{i}" for i in range(7)]
    database = ParetoDatabase(
        objectives, population_size=8, num_islands=1, selection="nsga3", random_seed=8
    )
    rng = np.random.default_rng(33)
    for i in range(12):
        database.add(program(str(i), rng.uniform(size=7)), i)
    resumed = ParetoDatabase.from_dict(database.to_dict())
    for _ in range(5):
        assert database.sample()[0].id == resumed.sample()[0].id
