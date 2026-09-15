"""Validate a released GeneBench Pro case through account Astra and its reference grader."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

from moevo.codex.client import require_chatgpt_login
from moevo.codex.container_runtime import solve_in_container
from moevo.codex.finance_pilot import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="wf_selection")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    source = Path("data/raw/genebench_pro").resolve()
    problem = source / "problems" / args.task
    config = json.loads((problem / "eval_config.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "genebench_reference", source / "reference_grader.py"
    )
    assert spec is not None and spec.loader is not None
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    positive = grader.evaluate(config, {"answer": config["ground_truth"]})
    negative = grader.evaluate(config, {"answer": {}})
    if not positive["passed"] or negative["passed"]:
        raise ValueError("Reference grader failed positive/negative controls")
    output = Path("results/domain_validation/genebench_pro") / args.task
    if (output / "report.json").exists():
        print((output / "report.json").read_text())
        return
    workspace = output.resolve() / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    for filename in config["data_files"]:
        origin = (problem / filename).resolve()
        if not origin.is_relative_to(problem.resolve() / "data_files"):
            raise ValueError("Unexpected input path")
        target = workspace / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
    version = require_chatgpt_login()
    response = solve_in_container(
        config["task"] + "\n\nInput files are available under /workspace/data_files. "
        "Use the benchmark run tool to analyze them. Return the requested JSON in your final answer.",
        workspace,
        output,
        timeout=args.timeout,
    )
    try:
        submission = json.loads(response.text)
        if not isinstance(submission, dict):
            submission = {}
    except json.JSONDecodeError:
        submission = {}
    scored = grader.evaluate(config, submission)
    report = {
        "dataset": "GeneBench Pro public (10-case release)",
        "task_id": args.task,
        "model": "gpt-6-astra",
        "effort": "xhigh",
        "cli_version": version,
        "source": json.loads((source / "_moevo_manifest.json").read_text()),
        "positive_control": positive,
        "negative_control": negative,
        "grade": scored,
        "e2e_smoke_passed": True,
        "headroom_confirmed": False,
        "full_benchmark_comparable": False,
        "usage": response.usage,
        "duration_s": response.duration_s,
        "response": response.text,
    }
    write_json(output / "report.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k not in {"source", "response"}}, indent=2)
    )


if __name__ == "__main__":
    main()
