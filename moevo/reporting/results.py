"""Export auditable percentage tables from a dashboard snapshot."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _percent(value):
    return None if value is None else 100 * value


def export_tables(data: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    aggregate = data["aggregate"]
    rows = []
    for level, items, key in (
        ("benchmark", aggregate["benchmarks"], "benchmark"),
        ("domain", aggregate["domains"], "domain"),
        ("overall", [{"name": "Equal-domain aggregate", **aggregate["overall"]}], "name"),
    ):
        for item in items:
            rows.append(
                {
                    "level": level,
                    "name": item[key],
                    "domain": item.get("domain", ""),
                    "metric": item.get("metric", "Mean normalized score"),
                    "n_tasks": item["n_tasks"],
                    "n_evaluations": item["n_evaluations"],
                    "baseline_percent": _percent(item["baseline"]),
                    "candidate_percent": _percent(item["candidate"]),
                    "delta_pp": item["delta_pp"],
                    "baseline_native_percent": _percent(item.get("baseline_native")),
                    "candidate_native_percent": _percent(item.get("candidate_native")),
                }
            )
    with (output / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["step", "candidate_sha256", "mean_percent", "minimum_percent"]
            + data["objectives"],
        )
        writer.writeheader()
        for point in data["history"]:
            writer.writerow(
                {
                    "step": point["step"],
                    "candidate_sha256": point["candidate_sha256"],
                    "mean_percent": 100 * point["mean"],
                    "minimum_percent": 100 * point["minimum"],
                    **{k: 100 * v for k, v in point["metrics"].items() if k in data["objectives"]},
                }
            )
    metadata = {
        "exported_utc": datetime.now(UTC).isoformat(),
        "status": data["status"],
        "purpose": data["purpose"],
        "split": data["split"],
        "protocol": data["protocol"],
        "solver": data["solver"],
        "judge": data["judge"],
        "authentication": "codex_chatgpt_account",
        "baseline_sha256": (data.get("baseline") or {}).get("candidate_sha256"),
        "candidate_sha256": (data.get("champion") or {}).get("candidate_sha256"),
        "candidate_selection": "Lexicographic maximization of sorted domain search scores",
        "weighting": aggregate["overall"]["weighting"],
        "task_attempts": data["task_evaluations"],
        "task_errors": data["task_errors"],
        "independent_heldout_results": False,
        "uncertainty": "Not estimated for the diagnostic fixture panel; repeated cache reads are not samples",
        "scores": rows,
    }
    (output / "results.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    def display(value, signed=False):
        return "—" if value is None else f"{value:+.1f}" if signed else f"{value:.1f}"

    caption = (
        "Diagnostic search-panel scores (%), with equal weight per domain in the aggregate. "
        "Each row uses the same fixed tasks for baseline and candidate. "
        "These tasks were available during evolution; these are not held-out benchmark results. "
        "Dashes indicate incomplete evaluations. Native metrics and scoring variants are in scores.csv."
    )
    md = [
        "# SuperHarness · diagnostic results",
        "",
        caption,
        "",
        "| Benchmark / aggregate | Tasks | Base Astra (%) | SuperHarness (%) | Δ (pp) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Benchmark / aggregate & $n$ tasks & Base Astra (\%) & SuperHarness (\%) & $\Delta$ (pp) \\",
        r"\hline",
    ]
    for row in rows:
        if row["level"] == "domain":
            continue
        values = [
            row["name"],
            str(row["n_tasks"]),
            display(row["baseline_percent"]),
            display(row["candidate_percent"]),
            display(row["delta_pp"], signed=True),
        ]
        md.append("| " + " | ".join(values) + " |")
        tex.append(" & ".join(v.replace("_", r"\_").replace("—", "--") for v in values) + r" \\")
    md.extend(["", "The aggregate is a normalized score, not a pooled benchmark accuracy.", ""])
    tex.extend(
        [
            r"\hline",
            r"\end{tabular}",
            "\\caption{" + caption.replace("%", r"\%") + "}",
            r"\end{table}",
        ]
    )
    (output / "results.md").write_text("\n".join(md))
    (output / "results.tex").write_text("\n".join(tex) + "\n")
