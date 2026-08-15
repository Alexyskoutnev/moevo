"""Loader for OpenAI's SimpleQA benchmark.

SimpleQA measures factual accuracy and calibration on obscure but verifiable
questions. It contains 4,326 short-answer factual questions where the correct
answer is a single, unambiguous entity that can be verified against reliable
sources. The questions are deliberately chosen to be difficult -- covering
niche topics in science, history, awards, geography, and more -- so that
models must either know the answer or honestly say "I don't know."

The key metric is not just accuracy but calibration: a well-calibrated agent
should have high accuracy on questions it answers confidently and should
abstain (say "I don't know") on questions it is uncertain about, rather than
confabulating a plausible-sounding but wrong answer.

Source: OpenAI (2024), "SimpleQA: A Benchmark for Short-Form Factuality."
Size: 4,326 questions (test split).
Data format: CSV with columns: problem (the question), answer (ground truth),
    metadata (topic/category string).

Real examples:

    Example 1:
        Q: "Who received the IEEE Frank Rosenblatt Award in 2010?"
        A: "Michio Sugeno"

    Example 2:
        Q: "Who was awarded the Oceanography Society's Jerlov Award in 2018?"
        A: "Annick Bricaud"

RSI relevance: SimpleQA detects whether self-improvement causes the agent to
become overconfident -- generating plausible-sounding answers to questions it
does not actually know. A well-calibrated agent should maintain or improve its
"I don't know" rate on genuinely hard questions. If an RSI iteration reduces
abstention without increasing accuracy, the agent has become less reliable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from moevo.data.loaders.base import BaseLoader
from moevo.data.types import BenchmarkType, DatasetInfo, Sample


class SimpleQALoader(BaseLoader):
    """Loads SimpleQA: 4,326 short factual questions testing calibration.

    Questions are deliberately obscure so that correct answers require genuine
    knowledge, not pattern matching. The benchmark rewards honest abstention
    over confident confabulation.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        default_dir = Path("data/raw/simpleqa")
        super().__init__(data_dir or default_dir)

    def load(self) -> list[Sample]:
        df = self._read_data()
        samples = []
        for idx, row in df.iterrows():
            # SimpleQA may have different column names depending on source.
            prompt = row.get("problem", row.get("question", row.get("prompt", "")))
            answer = row.get("answer", row.get("reference", ""))
            topic = row.get("metadata", row.get("topic", ""))
            samples.append(
                Sample(
                    id=f"simpleqa_{idx}",
                    benchmark="simpleqa",
                    benchmark_type=BenchmarkType.HALLUCINATION,
                    prompt=str(prompt),
                    reference=str(answer),
                    metadata={"topic": str(topic)},
                )
            )
        return samples

    def info(self) -> DatasetInfo:
        df = self._read_data()
        return DatasetInfo(
            name="SimpleQA",
            benchmark_type=BenchmarkType.HALLUCINATION,
            num_samples=len(df),
            columns=list(df.columns),
            splits_available=["test"],
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
