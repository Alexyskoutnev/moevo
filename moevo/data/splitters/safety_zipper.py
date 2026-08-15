"""Zipper schedule splitter for ToolEmu safety benchmark.

Creates deterministic, non-overlapping slices that align with the GDPval
zipper schedule (S1-S8 dev, E1-E2 eval).

ToolEmu: 144 samples across 36 toolkits → ~14 per slice (10 slices)

The split is deterministic (sorted by ID, round-robin) so every evolution
run evaluates on the exact same safety tasks per slice.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from moevo.data.types import Sample

NUM_DEV_SLICES = 8
NUM_EVAL_SLICES = 2
NUM_SLICES = NUM_DEV_SLICES + NUM_EVAL_SLICES


def safety_zipper_split(
    toolemu_samples: list[Sample],
) -> dict[str, dict[str, list[Sample]]]:
    """Split ToolEmu samples into 10 slices aligned with GDPval's S1-S8, E1-E2.

    Algorithm:
    1. Sort samples by ID for determinism.
    2. Round-robin assign across 10 slices.

    Returns:
        {"dev_slices": {"S1": [...], ...}, "eval_slices": {"E1": [...], ...}}
    """
    sorted_samples = sorted(toolemu_samples, key=lambda s: s.id)
    slices: list[list[Sample]] = [[] for _ in range(NUM_SLICES)]
    for i, s in enumerate(sorted_samples):
        slices[i % NUM_SLICES].append(s)

    dev_slices = {f"S{i + 1}": slices[i] for i in range(NUM_DEV_SLICES)}
    eval_slices = {f"E{i + 1}": slices[NUM_DEV_SLICES + i] for i in range(NUM_EVAL_SLICES)}

    return {"dev_slices": dev_slices, "eval_slices": eval_slices}


def save_safety_split(
    split: dict[str, dict[str, list[Sample]]],
    output_path: Path,
) -> None:
    """Save the safety zipper split to JSON for reproducibility."""
    data = {
        kind: {name: [s.id for s in samples] for name, samples in slices.items()}
        for kind, slices in split.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
