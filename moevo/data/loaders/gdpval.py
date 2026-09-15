"""Loader for the GDPval dataset (OpenAI).

GDPval measures whether AI agents can produce real economic value by completing
professional tasks that human workers are paid to do. It contains 220 tasks
spanning 44 occupations across 9 GDP-weighted sectors (Finance, Healthcare,
Technology, Education, etc.), with 5 tasks per occupation. Each task requires
producing concrete work deliverables -- Excel spreadsheets, Word documents,
PDFs -- that are scored against 50+ weighted rubric criteria by an LLM judge.

This is the primary capability benchmark in our RSI pipeline because it
directly measures the economic utility that self-improving agents must
maintain (or increase) across iterations. A drop in GDPval score means the
agent is producing worse work products, regardless of what other benchmarks
show.

Source: OpenAI (2025), "Measuring the economic value of AI" technical report.
Size: 220 tasks, 44 occupations, 9 sectors.
Data format: Parquet with columns: task_id, prompt, rubric_pretty, rubric_json,
    sector, occupation, reference_files, deliverable_files.

Record shape (content is not reproduced here -- GDPval is distributed by OpenAI
under its own terms and is downloaded at setup, not vendored in this repo):

    prompt      A multi-paragraph brief written in the voice of a colleague
                assigning work: a role, a business context, one or more
                attached reference files, and a requested deliverable.
    occupation  One of 44 occupations (e.g. Auditor, Finance Lead).
    sector      The GDPval sector label for the occupation.
    files       Reference material the agent must open (.xlsx, .docx, .pdf).
    rubric      A list of weighted criteria, each worth an integer number of
                points, checked against the produced deliverable. Criteria
                range from mechanical (the output is .xlsx, a named worksheet
                exists) to substantive (a figure is computed correctly).

See https://arxiv.org/abs/2510.04374 for the benchmark and its licence.

RSI relevance: GDPval is the "capability ceiling" metric. If recursive
self-improvement degrades the agent's ability to produce correct spreadsheets,
reports, and analyses, the improvement is not reliable -- it traded real-world
utility for something else. We track GDPval across every RSI iteration to
ensure economic value is never sacrificed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from moevo.data.loaders.base import BaseLoader
from moevo.data.types import BenchmarkType, DatasetInfo, Sample


class GDPvalLoader(BaseLoader):
    """Loads GDPval: 220 real-world professional tasks across 44 occupations.

    Each task simulates a paid professional role (auditor, data analyst, finance
    lead, etc.) and requires the agent to produce file-based deliverables that
    are graded against a detailed weighted rubric by a frozen LLM judge.
    """

    EXPECTED_TASKS = 220
    TASKS_PER_OCCUPATION = 5

    def __init__(self, data_dir: Path | None = None) -> None:
        default_dir = Path("data/raw/gdpval")
        super().__init__(data_dir or default_dir)

    def load(self) -> list[Sample]:
        df = self._read_parquet()
        samples = []
        for row in df.to_dict(orient="records"):
            samples.append(
                Sample(
                    id=row["task_id"],
                    benchmark="gdpval",
                    benchmark_type=BenchmarkType.ECONOMIC,
                    prompt=row["prompt"],
                    reference=row.get("rubric_pretty", ""),
                    metadata={
                        "sector": row.get("sector", ""),
                        "occupation": row.get("occupation", ""),
                        "rubric_json": row.get("rubric_json", ""),
                        "reference_files": row.get("reference_files", []),
                        "deliverable_files": row.get("deliverable_files", []),
                    },
                )
            )
        return samples

    def info(self) -> DatasetInfo:
        df = self._read_parquet()
        return DatasetInfo(
            name="GDPval",
            benchmark_type=BenchmarkType.ECONOMIC,
            num_samples=len(df),
            columns=list(df.columns),
            splits_available=["full"],
        )

    def _read_parquet(self) -> pd.DataFrame:
        parquet_files = list(self._data_dir.glob("*.parquet"))
        if parquet_files:
            return pd.read_parquet(parquet_files[0])
        # Hugging Face snapshots use data/train-*.parquet; older downloads
        # used train/*.parquet. Read every shard from the matching layout.
        for train_dir in (self._data_dir / "data", self._data_dir / "train"):
            parquet_files = sorted(train_dir.glob("*.parquet"))
            if parquet_files:
                return pd.concat(
                    [pd.read_parquet(path) for path in parquet_files], ignore_index=True
                )
        raise FileNotFoundError(
            f"No parquet files found in {self._data_dir}. Run the download script first."
        )

    def occupations(self) -> list[str]:
        """Return sorted list of unique occupations."""
        df = self._read_parquet()
        return sorted(df["occupation"].unique().tolist())

    def sectors(self) -> list[str]:
        """Return sorted list of unique sectors."""
        df = self._read_parquet()
        return sorted(df["sector"].unique().tolist())
