"""Central registry for all benchmark datasets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from moevo.data.loaders.agent_safety_bench import AgentSafetyBenchLoader
from moevo.data.loaders.agentharm import AgentHarmLoader
from moevo.data.loaders.asb import ASBLoader
from moevo.data.loaders.gdpval import GDPvalLoader
from moevo.data.loaders.harmbench import HarmBenchLoader
from moevo.data.loaders.ifeval import IFEvalLoader
from moevo.data.loaders.or_bench import ORBenchLoader
from moevo.data.loaders.simpleqa import SimpleQALoader
from moevo.data.loaders.toolemu import ToolEmuLoader
from moevo.data.loaders.truthfulqa import TruthfulQALoader

if TYPE_CHECKING:
    from moevo.data.loaders.base import BaseLoader
    from moevo.data.types import BenchmarkType, DatasetInfo, Sample

_LOADER_CLASSES: dict[str, type[BaseLoader]] = {
    "gdpval": GDPvalLoader,
    "truthfulqa": TruthfulQALoader,
    "simpleqa": SimpleQALoader,
    "ifeval": IFEvalLoader,
    "harmbench": HarmBenchLoader,
    "or_bench": ORBenchLoader,
    "agentharm": AgentHarmLoader,
    "toolemu": ToolEmuLoader,
    "asb": ASBLoader,
    "agent_safety_bench": AgentSafetyBenchLoader,
}


class DatasetRegistry:
    """Discovers and loads all available benchmark datasets.

    Usage:
        registry = DatasetRegistry(data_root=Path("data/raw"))
        registry.load_available()
        samples = registry.get_samples("gdpval")
        all_samples = registry.get_all_samples()
    """

    def __init__(self, data_root: Path | None = None) -> None:
        self._data_root = data_root or Path("data/raw")
        self._loaders: dict[str, BaseLoader] = {}
        self._available: list[str] = []

    def load_available(self) -> list[str]:
        """Discover and load all datasets that exist on disk.

        Returns:
            List of dataset names that were successfully loaded.
        """
        self._available = []
        for name, loader_cls in _LOADER_CLASSES.items():
            data_dir = self._data_root / name
            if data_dir.exists():
                try:
                    loader = loader_cls(data_dir)
                    loader.samples()  # Trigger load to verify data exists.
                    self._loaders[name] = loader
                    self._available.append(name)
                except FileNotFoundError:
                    continue
        return self._available

    def load_dataset(self, name: str) -> BaseLoader:
        """Load a specific dataset by name.

        Raises:
            KeyError: If the dataset name is not recognized.
            FileNotFoundError: If the data files are missing.
        """
        if name not in _LOADER_CLASSES:
            raise KeyError(f"Unknown dataset: {name}. Available: {list(_LOADER_CLASSES.keys())}")
        data_dir = self._data_root / name
        loader = _LOADER_CLASSES[name](data_dir)
        loader.samples()  # Trigger load.
        self._loaders[name] = loader
        if name not in self._available:
            self._available.append(name)
        return loader

    def get_samples(self, name: str) -> list[Sample]:
        """Get all samples for a specific dataset."""
        if name not in self._loaders:
            self.load_dataset(name)
        return self._loaders[name].samples()

    def get_all_samples(self) -> list[Sample]:
        """Get all samples across all loaded datasets."""
        all_samples = []
        for loader in self._loaders.values():
            all_samples.extend(loader.samples())
        return all_samples

    def get_samples_by_type(self, benchmark_type: BenchmarkType) -> list[Sample]:
        """Get all samples of a specific benchmark type."""
        return [s for s in self.get_all_samples() if s.benchmark_type == benchmark_type]

    def info(self) -> dict[str, DatasetInfo]:
        """Return metadata for all loaded datasets."""
        return {name: loader.info() for name, loader in self._loaders.items()}

    @property
    def available_datasets(self) -> list[str]:
        """Names of all successfully loaded datasets."""
        return list(self._available)

    @staticmethod
    def supported_datasets() -> list[str]:
        """Names of all datasets the registry knows how to load."""
        return list(_LOADER_CLASSES.keys())
