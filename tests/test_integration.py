"""Integration test: run 3 iterations with a trivial evaluator."""

from __future__ import annotations

import asyncio
import os

import pytest

from moevo.controller import MoevoController
from moevo.core import MoevoConfig

TRIVIAL_EVALUATOR = '''\
import random

def evaluate(program_path):
    """Trivial evaluator: random scores."""
    return {
        "score1": random.random(),
        "score2": random.random(),
    }
'''

SEED_PROGRAM = """\
def solve():
    return 42
"""


def _has_api_key() -> bool:
    return bool(
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create temp files for evaluator and seed program."""
    eval_path = tmp_path / "eval.py"
    eval_path.write_text(TRIVIAL_EVALUATOR)
    seed_path = tmp_path / "seed.py"
    seed_path.write_text(SEED_PROGRAM)
    output_dir = tmp_path / "output"
    return eval_path, seed_path, output_dir


@pytest.mark.skipif(
    not _has_api_key(),
    reason="No LLM API key available",
)
def test_three_iterations(tmp_workspace):
    """Run 3 iterations and verify loop completes with checkpoints."""
    eval_path, seed_path, output_dir = tmp_workspace

    config = MoevoConfig(
        initial_program=str(seed_path),
        evaluator_path=str(eval_path),
        objectives=["score1", "score2"],
        model="gemini/gemini-2.0-flash",
        iterations=3,
        population_size=4,
        num_islands=2,
        checkpoint_interval=1,
        output_dir=str(output_dir),
    )

    controller = MoevoController(config)
    result = asyncio.run(controller.run())

    assert result.iterations_completed == 3
    assert len(result.pareto_front) >= 1
    assert result.best_program is not None
    assert (output_dir / "best_program.py").exists()
    assert (output_dir / "pareto_front").exists()
    assert list(output_dir.glob("checkpoint_*.json"))
