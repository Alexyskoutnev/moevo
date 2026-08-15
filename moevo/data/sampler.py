"""Unified sampler for training and evaluation across all benchmarks."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moevo.data.registry import DatasetRegistry
    from moevo.data.types import BenchmarkType, Sample


class UnifiedSampler:
    """Clean interface to sample from any loaded benchmark.

    Supports sampling by benchmark name, type, or mixed batches
    with configurable proportions.

    Usage:
        registry = DatasetRegistry()
        registry.load_available()
        sampler = UnifiedSampler(registry, seed=42)

        # Sample from a specific benchmark.
        batch = sampler.sample("gdpval", n=10)

        # Sample from a benchmark type.
        batch = sampler.sample_by_type(BenchmarkType.HALLUCINATION, n=20)

        # Sample a mixed batch with proportions.
        batch = sampler.sample_mixed(n=100, weights={
            "gdpval": 0.5,
            "truthfulqa": 0.2,
            "ifeval": 0.15,
            "harmbench": 0.15,
        })
    """

    def __init__(self, registry: DatasetRegistry, seed: int = 42) -> None:
        self._registry = registry
        self._rng = random.Random(seed)
        self._index: dict[str, list[Sample]] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the internal index from the registry."""
        self._index = {}
        for name in self._registry.available_datasets:
            self._index[name] = self._registry.get_samples(name)

    def sample(self, benchmark: str, n: int) -> list[Sample]:
        """Sample n items from a specific benchmark.

        Args:
            benchmark: Name of the benchmark (e.g., "gdpval", "truthfulqa").
            n: Number of samples to return.

        Returns:
            List of samples. If n > available, returns all available.
        """
        pool = self._index.get(benchmark, [])
        if not pool:
            raise KeyError(
                f"Benchmark '{benchmark}' not loaded. Available: {list(self._index.keys())}"
            )
        n = min(n, len(pool))
        return self._rng.sample(pool, n)

    def sample_by_type(self, benchmark_type: BenchmarkType, n: int) -> list[Sample]:
        """Sample n items from all benchmarks of a given type.

        Pools all benchmarks of the specified type and samples uniformly.
        """
        pool = [
            s
            for samples in self._index.values()
            for s in samples
            if s.benchmark_type == benchmark_type
        ]
        if not pool:
            raise KeyError(f"No loaded benchmarks of type: {benchmark_type.value}")
        n = min(n, len(pool))
        return self._rng.sample(pool, n)

    def sample_mixed(
        self,
        n: int,
        weights: dict[str, float] | None = None,
    ) -> list[Sample]:
        """Sample a mixed batch from multiple benchmarks.

        Args:
            n: Total number of samples.
            weights: Dict mapping benchmark name to proportion (must sum to ~1.0).
                     If None, samples uniformly across all loaded benchmarks.

        Returns:
            Mixed list of samples from multiple benchmarks.
        """
        if weights is None:
            # Uniform across loaded benchmarks.
            available = list(self._index.keys())
            weights = {name: 1.0 / len(available) for name in available}

        batch: list[Sample] = []
        remaining = n
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        for i, (name, weight) in enumerate(sorted_weights):
            if name not in self._index:
                continue
            # Last benchmark gets whatever is remaining.
            count = remaining if i == len(sorted_weights) - 1 else round(n * weight)
            count = min(count, remaining, len(self._index[name]))
            if count > 0:
                batch.extend(self._rng.sample(self._index[name], count))
                remaining -= count

        self._rng.shuffle(batch)
        return batch

    def get_all(self, benchmark: str) -> list[Sample]:
        """Return all samples from a specific benchmark (no sampling)."""
        if benchmark not in self._index:
            raise KeyError(f"Benchmark '{benchmark}' not loaded.")
        return list(self._index[benchmark])

    def stats(self) -> dict[str, int]:
        """Return sample counts per loaded benchmark."""
        return {name: len(samples) for name, samples in self._index.items()}

    def stats_by_type(self) -> dict[str, int]:
        """Return sample counts grouped by benchmark type."""
        counts: dict[str, int] = defaultdict(int)
        for samples in self._index.values():
            if samples:
                counts[samples[0].benchmark_type.value] += len(samples)
        return dict(counts)
