"""JSON checkpoint save/load for moevo runs."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from ..search.database import ParetoDatabase

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("moevo.checkpoint")


def save_checkpoint(db: ParetoDatabase, iteration: int, output_dir: Path) -> Path:
    """Save database state to a JSON checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_{iteration:04d}.json"

    data = {
        "iteration": iteration,
        "database": db.to_dict(),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("Checkpoint saved: %s", path)
    return path


def load_checkpoint(path: Path) -> tuple[ParetoDatabase, int]:
    """Load database state from a JSON checkpoint.

    Returns (database, iteration).
    """
    with open(path) as f:
        data = json.load(f)

    db = ParetoDatabase.from_dict(data["database"])
    iteration = data["iteration"]
    logger.info("Checkpoint loaded: %s (iteration %d)", path, iteration)
    return db, iteration


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the most recent checkpoint file in the output directory."""
    if not output_dir.exists():
        return None
    checkpoints = sorted(output_dir.glob("checkpoint_*.json"))
    return checkpoints[-1] if checkpoints else None
