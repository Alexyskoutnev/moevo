"""Build and validate a local SuperHarness submission; never upload it.

Dummy mode sends deliberately empty predictions through actual reference graders.
Smoke mode runs the same frozen policy through account Astra for two development
tasks. Neither mode is a leaderboard submission or a headroom measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import zipfile
from pathlib import Path

from experiments.validate_domains import IMAGE, offline_grade, tasks
from moevo.codex.client import require_chatgpt_login
from moevo.codex.container_runtime import solve_in_container
from moevo.codex.finance_pilot import load_data, score_answer, task_prompt, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBMISSION = ROOT / "submissions/superharness-dummy/submission.json"
CATALOG = {
    "finqa": ("Finance", "FinQA"),
    "bizfinbench2": ("Finance", "BizFinBench.v2"),
    "genebench_pro": ("Science", "GeneBench Pro public"),
    "terminal_bench_science": ("Science", "Terminal-Bench Science"),
    "amo": ("Mathematics", "AMO-Bench P subset"),
    "putnambench": ("Mathematics", "PutnamBench"),
    "travelplanner": ("Planning", "TravelPlanner"),
    "oragentbench": ("Operations research", "ORAgentBench"),
    "dsbench": ("Data analysis", "DSBench"),
    "gdpval": ("Professional work", "GDPval"),
    "automationbench": ("Professional work", "AutomationBench public"),
    "terminal_bench_2": ("Software / terminal", "Terminal-Bench 2"),
    "harvey_lab": ("Legal", "Harvey LAB"),
    "healthbench_professional": ("Medicine", "HealthBench Professional"),
    "tau3_bench": ("Customer service", "tau3-bench"),
    "eduagentbench": ("Education", "EduAgentBench"),
}
SUPPORTED = {"finqa", "amo"}


def load_submission(path: Path) -> dict:
    from moevo.codex.judging import judge_options

    submission = json.loads(path.read_text())
    judge_options(submission)
    fixed = {
        "schema_version": 1,
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "authentication": "codex_chatgpt_account",
    }
    for field, value in fixed.items():
        if submission.get(field) != value:
            raise ValueError(f"Submission must set {field} to {value!r}")
    for field in ("submission_id", "description", "instructions"):
        if not isinstance(submission.get(field), str) or not submission[field].strip():
            raise ValueError(f"Submission needs nonempty {field}")
    for field, low, high in (("timeout_seconds", 60, 1800), ("max_tool_calls", 1, 80)):
        if type(submission.get(field)) is not int or not low <= submission[field] <= high:
            raise ValueError(f"Invalid {field}")
    if type(submission.get("seed")) is not int:
        raise ValueError("Submission seed must be an integer")
    benchmarks = submission.get("benchmarks")
    if (
        not isinstance(benchmarks, list)
        or not benchmarks
        or any(not isinstance(name, str) or name not in CATALOG for name in benchmarks)
    ):
        raise ValueError("Unknown or empty benchmark inventory")
    if len(set(benchmarks)) != len(benchmarks):
        raise ValueError("Duplicate benchmarks")
    return submission


def fixture(name: str, seed: int, output: Path, image: str):
    """Return only the public prompt to the solver; retain labels in the grader."""
    if name == "finqa":
        train, _, official = load_data(ROOT / "data/raw/finqa")
        task_id = random.Random(seed).sample(sorted(train), 128)[0]
        row = train[task_id]

        def grade(prediction: str, label: str) -> dict:
            score, feedback = score_answer(prediction, row, official)
            result = {"score": score, "feedback": feedback}
            write_json(output / label / "grade.json", result)
            return result

        gold = json.dumps({"program": official["program_tokenization"](row["qa"]["program"])})
        negative = '{"program": []}'
        prompt = task_prompt(row)
        source = ROOT / "data/raw/finqa/manifest.json"
        split = "train"
    else:
        rows = tasks("amo")
        index = random.Random(seed).sample(range(len(rows)), 1)[0]
        row = rows[index]
        task_id = f"p-subset-{index:04d}"

        def grade(prediction: str, label: str) -> dict:
            return offline_grade("amo", row, prediction, output / label, image)

        gold, negative, prompt = row["answer"], "No answer submitted", row["prompt"]
        source = ROOT / "data/raw/amo_bench/_moevo_manifest.json"
        split = "public development fixture; not a held-out score"
    positive_grade = grade(gold, "positive_control")
    negative_grade = grade(negative, "negative_control")
    if positive_grade["score"] != 1 or negative_grade["score"] != 0:
        raise ValueError("Reference grader failed positive/negative controls")
    metadata = {
        "task_id": task_id,
        "split": split,
        "data_revision": json.loads(source.read_text())["revision"],
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "controls": {
            "positive_score": positive_grade["score"],
            "negative_score": negative_grade["score"],
        },
    }
    return prompt, negative, grade, metadata


def write_bundle(output: Path, submission: dict, records: list[dict], mode: str, provenance: dict):
    """Package explicit results without raw tasks, gold answers, or credentials."""
    summary = {
        "submission_id": submission["submission_id"],
        "mode": mode,
        "dummy": mode == "dummy",
        "evolved": False,
        "uploaded": False,
        "full_benchmark_evaluation": False,
        "eligible_for_evolution": False,
        "provenance": provenance,
        "tasks_scored": sum(row["status"] in {"dummy_scored", "scored"} for row in records),
        "infrastructure_errors": max(
            sum(row["status"] == "error" for row in records),
            int("infrastructure_error" in provenance),
        ),
        "results": records,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "submission.json", submission)
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in records)
    )
    readme = (
        f"# {submission['submission_id']}\n\nMode: **{mode}**. Local submission format v1.\n\n"
        "This is a pipeline smoke test. Dummy predictions are deliberately empty and their "
        "scores are not model performance. Missing scores are null, never zero. "
        "Live smoke results cover one development task per selected adapter, not a benchmark.\n\n"
        "The frozen shared policy is submission.json. Run it from the Moevo checkout with "
        "`python experiments/run_submission.py --submission PATH --mode dummy|smoke --output NEW_DIR`. "
        "Dataset downloads and the pinned local Docker runtime are required. No upload occurred.\n"
    )
    (output / "README.md").write_text(readme)
    members = ["submission.json", "summary.json", "predictions.jsonl", "README.md"]
    hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in members}
    write_json(output / "checksums.json", hashes)
    with zipfile.ZipFile(output / "submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in [*members, "checksums.json"]:
            archive.write(output / name, name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--mode", choices=["dummy", "smoke"], default="dummy")
    parser.add_argument("--benchmarks", nargs="+", choices=list(CATALOG), default=["finqa", "amo"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    submission = load_submission(args.submission)
    selected = set(args.benchmarks)
    if not selected.issubset(submission["benchmarks"]):
        parser.error("Selected benchmarks are absent from the submission")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty output directory; existing runs are preserved")
    output.mkdir(parents=True, exist_ok=True)
    image = IMAGE
    provenance = {
        "policy_sha256": hashlib.sha256(
            json.dumps(submission, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "implementation_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                ROOT / "experiments/validate_domains.py",
                ROOT / "experiments/runtime/grade_domain.py",
                ROOT / "moevo/codex/client.py",
                ROOT / "moevo/codex/container_runtime.py",
                ROOT / "moevo/codex/container_server.py",
                ROOT / "moevo/codex/finance_pilot.py",
            ]
        },
    }
    records = [
        {
            "benchmark": name,
            "domain": CATALOG[name][0],
            "name": CATALOG[name][1],
            "status": "pending" if name in selected & SUPPORTED else "not_run",
            "score": None,
            "prediction": None,
            "reason": None
            if name in selected & SUPPORTED
            else (
                "Adapter not connected to this submission runner"
                if name in selected
                else "Outside this smoke run"
            ),
        }
        for name in submission["benchmarks"]
    ]
    try:
        if args.mode == "smoke" or "amo" in selected:
            inspected = subprocess.check_output(
                ["docker", "image", "inspect", IMAGE], text=True, timeout=30
            )
            image = json.loads(inspected)[0]["Id"]
            provenance["image_id"] = image
        if args.mode == "smoke":
            provenance["cli_version"] = require_chatgpt_login()
        for row in records:
            if row["status"] != "pending":
                continue
            task_output = output / "tasks" / row["benchmark"]
            try:
                prompt, negative, grade, metadata = fixture(
                    row["benchmark"], submission["seed"], task_output, image
                )
                row.update(metadata)
                prediction = negative
                if args.mode == "smoke":
                    response = solve_in_container(
                        submission["instructions"] + "\n\n" + prompt,
                        task_output / "workspace",
                        task_output / "agent",
                        image=image,
                        timeout=submission["timeout_seconds"],
                        max_tools=submission["max_tool_calls"],
                    )
                    prediction = response.text
                    row.update({"usage": response.usage, "duration_s": response.duration_s})
                result = grade(prediction, "prediction")
                row.update(
                    {
                        "status": "dummy_scored" if args.mode == "dummy" else "scored",
                        "score": result["score"],
                        "prediction": prediction,
                        "grade": result,
                    }
                )
                print(json.dumps({k: row[k] for k in ("benchmark", "status", "score")}), flush=True)
            except Exception as exc:
                row.update({"status": "error", "score": None, "reason": str(exc)})
                raise
    except Exception as exc:
        provenance["infrastructure_error"] = str(exc)
        for row in records:
            if row["status"] == "pending":
                row.update(
                    {"status": "not_run", "reason": "Run stopped after infrastructure error"}
                )
        write_bundle(output, submission, records, args.mode, provenance)
        raise
    summary = write_bundle(output, submission, records, args.mode, provenance)
    print(
        json.dumps(
            {"bundle": str(output / "submission.zip"), "tasks_scored": summary["tasks_scored"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
