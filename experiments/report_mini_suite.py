"""Merge mini-run shards into a publishable metadata-only validation report."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from experiments.run_submission import CATALOG
from moevo.codex.finance_pilot import write_json

ROOT = Path(__file__).resolve().parents[1]
DATA_NAMES = {"amo": "amo_bench"}


def merge(shards: list[Path]) -> dict:
    summaries = [json.loads((p / "summary.json").read_text()) for p in shards]
    signature = summaries[0]["policy_sha256"]
    if any(s["policy_sha256"] != signature or s["dummy"] for s in summaries):
        raise ValueError("All shards must be real runs with an identical frozen policy")
    rows = {}
    keep = {
        "benchmark",
        "domain",
        "name",
        "status",
        "score",
        "e2e_validated",
        "task_id",
        "protocol",
        "adapter_sha256",
        "runtime_image_id",
        "agent_image_id",
        "verifier_image_id",
        "strict_pass",
        "criterion_pass_rate",
        "artifact_count",
        "error",
        "validation_note",
        "controls_scope",
        "control_scope",
        "positive_control_validated",
        "termination",
        "from_scratch_replay_validated",
        "visual_layout_evaluated",
        "task_sha256",
    }
    for shard, summary in zip(shards, summaries, strict=True):
        for record in summary["results"]:
            if record["status"] == "not_run":
                continue
            name = str(record["benchmark"])
            if name in rows:
                raise ValueError(
                    f"Duplicate benchmark across shards: {name}; select one attempt explicitly"
                )
            if record["status"] == "running":
                raise ValueError(f"Benchmark is still running: {name}")
            row = {k: v for k, v in record.items() if k in keep}
            if "runtime_image_id" in row:
                row["base_preflight_image_id"] = row.pop("runtime_image_id")
            grade = record.get("grade", {})
            row["grade_summary"] = {
                k: v for k, v in grade.items() if isinstance(v, (int, float, bool))
            }
            if "action_checks" in grade:
                row["grade_summary"]["actions_matched"] = sum(
                    bool(a["action_match"]) for a in grade["action_checks"]
                )
                row["grade_summary"]["actions_total"] = len(grade["action_checks"])
            if "db_check" in grade:
                row["grade_summary"]["db_match"] = grade["db_check"]["db_match"]
            full_report = shard / name / "report.json"
            row["private_evidence"] = str(full_report.relative_to(ROOT))
            row["private_report_sha256"] = hashlib.sha256(full_report.read_bytes()).hexdigest()
            attempt = shard / record["attempt"]
            runtimes = []
            for path in sorted(attempt.rglob("runtime.json")):
                runtime = json.loads(path.read_text())
                runtimes.append({"path": str(path.relative_to(ROOT)), **runtime})
            row["recorded_agent_runtimes"] = runtimes
            trace_path = attempt / "agent/tool_trace.json"
            if trace_path.exists():
                trace = json.loads(trace_path.read_text())
                row["task_tool_calls"] = len(trace)
                if (
                    name == "terminal_bench_science"
                    and len(trace) >= 80
                    and not any((attempt / "workspace/results").glob("*"))
                    and any(".b64" in call.get("command", "") for call in trace)
                ):
                    row["validation_note"] = (
                        row.get("protocol", "")
                        + ". The attempt reached its 80-call budget while reading image base64 through "
                        "the text-only terminal and produced no required result files. No image-display "
                        "tool was enabled; this is an explicit limitation of this harness configuration."
                    )
            audit_path = attempt / "assertion_audit.json"
            if audit_path.exists():
                row["postrun_assertion_audit"] = json.loads(audit_path.read_text())
                row["validation_note"] = (
                    "Real task and native grader ran; no-op/synthetic-positive controls passed. "
                    "Fixed assertion IDs disagree with generated tool IDs, so task validity remains under audit."
                )
            if row.get("error"):
                row["error"] = row["error"].replace(str(ROOT), "<repo>")
            manifests = []
            for parent in (ROOT / "data/raw", ROOT / "data/external"):
                directory = parent / DATA_NAMES.get(name, name)
                for filename in (
                    "manifest.json",
                    "source_manifest.json",
                    "_moevo_manifest.json",
                    "_moevo_assets_manifest.json",
                ):
                    path = directory / filename
                    if path.exists():
                        data = json.loads(path.read_text())
                        manifests.append(
                            {
                                "path": str(path.relative_to(ROOT)),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                **{
                                    k: v
                                    for k, v in data.items()
                                    if k
                                    in {
                                        "dataset",
                                        "repository",
                                        "revision",
                                        "archive_sha256",
                                        "source",
                                        "url",
                                    }
                                },
                            }
                        )
            row["source_manifests"] = manifests
            rows[name] = row
    if set(rows) != set(CATALOG):
        raise ValueError(f"Missing benchmark status: {sorted(set(CATALOG) - set(rows))}")
    return {
        "purpose": "One real task per benchmark where runtime is available; integration validation only",
        "model": summaries[0]["model"],
        "effort": summaries[0]["effort"],
        "authentication": "Signed-in Codex ChatGPT account; API-key fallback disabled",
        "cli_version": summaries[0]["cli_version"],
        "policy_sha256": signature,
        "generated_utc": datetime.now(UTC).isoformat(),
        "dummy": False,
        "evolved": False,
        "full_benchmark_scores": False,
        "headroom_confirmed": False,
        "validated": sum(r["e2e_validated"] for r in rows.values()),
        "real_tasks_scored": sum(r["status"] in {"scored", "needs_audit"} for r in rows.values()),
        "results": [rows[name] for name in CATALOG],
    }


def markdown(report: dict) -> str:
    lines = [
        "# Real Astra mini-run",
        "",
        f"**{report['validated']}/16 benchmark integrations validated; {report['real_tasks_scored']} real task attempts scored.**",
        "",
        "One task per benchmark with an available runtime. These are integration results, not full-benchmark accuracy or evidence of saturation.",
        "",
        f"Model: `{report['model']}` / `{report['effort']}`. {report['authentication']}. No evolution was run.",
        "",
        "| Domain | Benchmark | Status | One-task native score |",
        "| --- | --- | --- | ---: |",
    ]
    for row in report["results"]:
        score = "—" if row["score"] is None else f"{row['score']:.4g}"
        details = row.get("grade_summary", {})
        if "n_criteria" in details:
            score += f" ({details['n_passed']}/{details['n_criteria']} criteria)"
        lines.append(f"| {row['domain']} | {row['name']} | {row['status']} | {score} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "`scored` means the model attempted a real task and the evaluation path passed its controls. A score of zero can still validate an integration. `needs_audit` retains a real score with a known protocol/data issue. `blocked` has no model score; setup or evaluator failure is never counted as model failure.",
        "",
        "The same shared instructions were used throughout. Task-specific tools and graders differ by benchmark. Positive/negative controls verify the evaluation path; they do not establish expert-level judge reliability.",
        "",
        "## Protocol qualifications",
        "",
    ]
    for row in report["results"]:
        note = row.get("error") or row.get("validation_note") or row.get("protocol", "")
        lines.append(f"- **{row['name']}:** {note}")
    lines += [
        "",
        "## Reproduction and evidence",
        "",
        "See the [run manifest and commands](../submissions/superharness-astra-v1/README.md) and [metadata-only report](../submissions/superharness-astra-v1/validation.json). The JSON records task identifiers, native metrics, source revisions, policy hash, and available runtime hashes. Raw model transcripts, submitted artifacts, grader rubrics, and downloaded benchmark data remain local under `results/` and `data/`.",
        "",
        "The larger medical screen was stopped when the task was narrowed to this mini run. None of these single-task results admits a benchmark as having demonstrated useful headroom. A separate, predeclared multi-task screen is still required before harness evolution.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    report = merge([p.resolve() for p in args.shards])
    write_json(ROOT / "submissions/superharness-astra-v1/validation.json", report)
    (ROOT / "docs/mini-run-status.md").write_text(markdown(report))
    print(json.dumps({k: report[k] for k in ("validated", "real_tasks_scored")}))


if __name__ == "__main__":
    main()
