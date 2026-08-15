"""Shared type definitions for the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Split(Enum):
    """Dataset split identifiers."""

    TRAIN = "train"
    TEST = "test"
    DEV = "dev"
    VAULT = "vault"  # Held-out test vault (used once at the end).


class BenchmarkType(Enum):
    """Categories of benchmarks in the RSI pipeline."""

    ECONOMIC = "economic"  # GDPval — real-world professional tasks.
    HALLUCINATION = "hallucination"  # TruthfulQA, SimpleQA.
    INSTRUCTION = "instruction"  # IFEval — constraint following.
    SAFETY = "safety"  # HarmBench, OR-Bench.


@dataclass(frozen=True)
class Sample:
    """A single sample from any benchmark.

    Unified representation across all dataset types. Each loader maps
    its native format into this structure.
    """

    id: str
    benchmark: str
    benchmark_type: BenchmarkType
    prompt: str
    reference: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata about a loaded dataset."""

    name: str
    benchmark_type: BenchmarkType
    num_samples: int
    columns: list[str]
    splits_available: list[str]
