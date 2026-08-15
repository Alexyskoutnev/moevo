"""Base class for dataset loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from moevo.data.types import DatasetInfo, Sample


class BaseLoader(ABC):
    """Abstract base for all dataset loaders.

    Each loader is responsible for:
    1. Loading raw data from disk (parquet/json/csv).
    2. Converting each row into a unified `Sample`.
    3. Reporting dataset metadata via `info()`.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._samples: list[Sample] = []
        self._loaded = False

    @property
    def name(self) -> str:
        """Human-readable dataset name."""
        return self.__class__.__name__.replace("Loader", "")

    @abstractmethod
    def load(self) -> list[Sample]:
        """Load and return all samples from the dataset."""

    @abstractmethod
    def info(self) -> DatasetInfo:
        """Return metadata about this dataset."""

    def samples(self) -> list[Sample]:
        """Return cached samples, loading if necessary."""
        if not self._loaded:
            self._samples = self.load()
            self._loaded = True
        return self._samples

    def _resolve_path(self, *parts: str) -> Path:
        """Resolve a path relative to the data directory."""
        path = self._data_dir.joinpath(*parts)
        if not path.exists():
            raise FileNotFoundError(f"Data not found: {path}")
        return path
