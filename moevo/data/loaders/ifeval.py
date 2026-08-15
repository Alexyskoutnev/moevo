"""Loader for Google's IFEval benchmark.

IFEval (Instruction-Following Eval) measures whether agents can precisely
follow verifiable format constraints embedded in natural language prompts.
It contains 541 prompts, each with one or more machine-checkable constraints
such as word count limits, forbidden characters, required formatting (markdown
headers, bullet points), language requirements, and structural rules.

Unlike subjective quality benchmarks, IFEval constraints are binary: the output
either satisfies "write at least 300 words with no commas" or it does not. This
makes it a clean, automated test of instruction adherence without requiring an
LLM judge.

Source: Zhou et al. (2023), "Instruction-Following Evaluation for Large
    Language Models", Google Research.
Size: 541 prompts (train split, used as test in practice).
Data format: JSON/Parquet with columns: key, prompt, instruction_id_list,
    kwargs (parameters for each constraint).

Real examples:

    Example 1:
        Prompt: "Write a 300+ word summary of the wikipedia page... Do not use
        any commas and highlight at least 3 sections in markdown format"
        Constraints: [length_constraints:number_words, punctuation:no_comma,
                      detectable_format:number_highlighted_sections]

    Example 2:
        Prompt: "Write an itinerary for Japan in a Shakespearean style. You are
        not allowed to use any commas"
        Constraints: [punctuation:no_comma]

RSI relevance: IFEval detects whether self-improvement degrades precise
instruction following. An agent that becomes more creative or capable but
starts ignoring explicit constraints (word counts, formatting rules, forbidden
characters) has become less controllable. In the RSI context, controllability
is a prerequisite for safe self-improvement -- an agent that cannot follow
precise instructions cannot reliably follow safety constraints either.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from moevo.data.loaders.base import BaseLoader
from moevo.data.types import BenchmarkType, DatasetInfo, Sample


class IFEvalLoader(BaseLoader):
    """Loads IFEval: 541 prompts with verifiable instruction constraints.

    Each prompt contains one or more machine-checkable format constraints
    (word count, forbidden punctuation, required sections, etc.) that can be
    verified programmatically without an LLM judge.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        default_dir = Path("data/raw/ifeval")
        super().__init__(data_dir or default_dir)

    def load(self) -> list[Sample]:
        df = self._read_data()
        samples = []
        for idx, row in df.iterrows():
            prompt = row.get("prompt", row.get("instruction", ""))
            samples.append(
                Sample(
                    id=f"ifeval_{row.get('key', idx)}",
                    benchmark="ifeval",
                    benchmark_type=BenchmarkType.INSTRUCTION,
                    prompt=str(prompt),
                    reference="",
                    metadata={
                        "instruction_id_list": row.get("instruction_id_list", []),
                        "kwargs": row.get("kwargs", []),
                    },
                )
            )
        return samples

    def info(self) -> DatasetInfo:
        df = self._read_data()
        return DatasetInfo(
            name="IFEval",
            benchmark_type=BenchmarkType.INSTRUCTION,
            num_samples=len(df),
            columns=list(df.columns),
            splits_available=["train"],
        )

    def _read_data(self) -> pd.DataFrame:
        for ext in ["*.parquet", "*.csv", "*.json", "*.jsonl"]:
            files = list(self._data_dir.rglob(ext))
            if files:
                path = files[0]
                if path.suffix == ".parquet":
                    return pd.read_parquet(path)
                elif path.suffix == ".csv":
                    return pd.read_csv(path)
                elif path.suffix in (".json", ".jsonl"):
                    return pd.read_json(path, lines=path.suffix == ".jsonl")
        raise FileNotFoundError(
            f"No data files found in {self._data_dir}. Run the download script first."
        )
