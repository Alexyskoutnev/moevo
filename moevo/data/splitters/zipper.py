"""Zipper schedule splitter for GDPval.

Splits the 220-task GDPval dataset into 10 slices of 22 tasks each:
  - 8 dev slices  (S1-S8): for iterative development and tuning.
  - 2 eval slices (E1-E2): held-out evaluation, run only at the end.

Tasks are assigned round-robin by occupation so every slice gets a
representative mix of occupations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from moevo.data.types import Sample

NUM_DEV_SLICES = 8
NUM_EVAL_SLICES = 2
NUM_SLICES = NUM_DEV_SLICES + NUM_EVAL_SLICES
TASKS_PER_SLICE = 22


@dataclass(frozen=True)
class ZipperSplit:
    """Result of the zipper split assignment."""

    dev_slices: dict[str, list[Sample]]  # S1..S8, 22 tasks each.
    eval_slices: dict[str, list[Sample]]  # E1..E2, 22 tasks each.


def zipper_split(samples: list[Sample]) -> ZipperSplit:
    """Split GDPval into 8 dev slices + 2 eval slices (22 tasks each).

    The split is fully deterministic (no randomness) so all agents
    are evaluated on the exact same set of tasks per slice.

    Algorithm:
    1. Group samples by occupation (44 occupations, 5 tasks each).
    2. Sort occupations alphabetically.
    3. Sort tasks within each occupation by task ID.
    4. Flatten into a single ordered list (occupation-major order).
    5. Round-robin assign tasks across 10 slices so each slice gets
       a representative mix of occupations.
    6. First 8 slices are dev (S1-S8), last 2 are eval (E1-E2).

    Args:
        samples: List of GDPval samples (must be exactly 220).

    Returns:
        ZipperSplit with dev_slices (S1-S8) and eval_slices (E1-E2).
    """
    # Group by occupation.
    by_occupation: dict[str, list[Sample]] = {}
    for s in samples:
        occ = s.metadata.get("occupation", "unknown")
        by_occupation.setdefault(occ, []).append(s)

    occupations = sorted(by_occupation.keys())
    if len(occupations) != 44:
        raise ValueError(
            f"Expected 44 occupations, got {len(occupations)}. Is this the full GDPval dataset?"
        )

    # Sort tasks within each occupation by task_id for determinism.
    for occ in occupations:
        by_occupation[occ].sort(key=lambda s: s.id)

    # Flatten: iterate occupations in order, tasks within each in order.
    ordered: list[Sample] = []
    for occ in occupations:
        ordered.extend(by_occupation[occ])

    assert len(ordered) == 220, f"Expected 220 tasks, got {len(ordered)}"

    # Round-robin into 10 slices.
    slices: list[list[Sample]] = [[] for _ in range(NUM_SLICES)]
    for i, sample in enumerate(ordered):
        slices[i % NUM_SLICES].append(sample)

    # Validate all slices are 22.
    for idx, sl in enumerate(slices):
        assert len(sl) == TASKS_PER_SLICE, (
            f"Slice {idx} has {len(sl)} tasks, expected {TASKS_PER_SLICE}"
        )

    # Name them: S1-S8 for dev, E1-E2 for eval.
    dev_slices = {f"S{i + 1}": slices[i] for i in range(NUM_DEV_SLICES)}
    eval_slices = {f"E{i + 1}": slices[NUM_DEV_SLICES + i] for i in range(NUM_EVAL_SLICES)}

    return ZipperSplit(dev_slices=dev_slices, eval_slices=eval_slices)


def save_split(split: ZipperSplit, output_path: Path) -> None:
    """Save the zipper split assignment to JSON for reproducibility."""
    data = {
        "dev_slices": {name: [s.id for s in samples] for name, samples in split.dev_slices.items()},
        "eval_slices": {
            name: [s.id for s in samples] for name, samples in split.eval_slices.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
