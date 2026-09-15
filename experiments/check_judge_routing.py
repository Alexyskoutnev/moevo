"""Run real account judge controls without re-running any benchmark solvers."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

from experiments.run_submission import load_submission
from experiments.validate_domains import ds_grade
from moevo.codex.client import require_chatgpt_login
from moevo.codex.document_tasks import gdp_grade, legal_verdict
from moevo.codex.finance_pilot import write_json
from moevo.codex.health_pilot import controls as health_controls
from moevo.codex.judging import judge_metadata

ROOT = Path(__file__).resolve().parents[1]


def run_controls(name: str, policy: dict, output: Path) -> dict:
    directory = output / name
    directory.mkdir(parents=True)
    checks = []
    if name == "healthbench":
        health_controls(directory, {"purpose": "judge-routing-controls"}, judge=policy)
        checks = json.loads((directory / "controls.json").read_text())["cases"]
    else:
        for label, text, expected in [
            ("positive", "2 and 3", True),
            ("negative", "No answer", False),
        ]:
            target = directory / label
            if name == "dsbench":
                grade = ds_grade(
                    "Return the numbers 2 and 3.", "2 and 3", text, target, judge=policy
                )
                passed = grade["score"] == float(expected)
            elif name == "gdpval":
                grade = gdp_grade(
                    "Return the numbers 2 and 3.",
                    [{"filename": "answer.txt", "text": text}],
                    [{"score": 2, "criterion": "The file contains both 2 and 3."}],
                    target,
                    judge=policy,
                )
                passed = grade["score"] == float(expected)
            else:
                grade = legal_verdict(
                    "Return the numbers 2 and 3.",
                    text,
                    "Numbers",
                    "PASS if the response includes both 2 and 3; otherwise FAIL.",
                    target,
                    judge=policy,
                )
                passed = grade["verdict"] == ("pass" if expected else "fail")
            checks.append({"name": label, "passed": passed, "grade": grade})
            if not passed:
                raise ValueError(f"{name}: {label} control failed")
    return {"judge": name, "status": "passed", "controls": len(checks), **judge_metadata(policy)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "submissions/superharness-astra-terra-pilot/submission.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = load_submission(args.submission)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    version = require_chatgpt_login()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = {
            pool.submit(run_controls, name, policy, args.output): name
            for name in ("dsbench", "gdpval", "harvey_lab", "healthbench")
        }
        for future in concurrent.futures.as_completed(pending):
            try:
                row = future.result()
            except Exception as exc:
                row = {"judge": pending[future], "status": "failed", "error": str(exc)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "purpose": "Real judge transport/polarity controls, not benchmark accuracy calibration",
        "solver_model": policy["model"],
        "solver_calls": 0,
        **judge_metadata(policy),
        "authentication": "codex_chatgpt_account",
        "cli_version": version,
        "policy_sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest(),
        "results": rows,
        "passed": all(r["status"] == "passed" for r in rows),
    }
    write_json(args.output / "summary.json", report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
