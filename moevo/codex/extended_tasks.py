"""Additional real-task adapters for the all-benchmark mini validation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from moevo.codex.client import account_environment
from moevo.codex.document_tasks import gdp, legal
from moevo.codex.domain_tasks import ROOT, result, solve
from moevo.codex.finance_pilot import write_json
from moevo.codex.terminal_tasks import cilia, regex


def operations(policy: dict, output: Path, image: str):
    from experiments.validate_oragentbench import grade

    task = ROOT / "data/external/oragentbench/harbor_tasks/industrial_water_reuse_blending"
    public = task / "environment/app"
    oracle = output / "oracle"
    shutil.copytree(public, oracle)
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--memory=4g",
            "--cpus=2",
            "--tmpfs",
            "/tmp:rw,size=512m",
            "--mount",
            f"type=bind,source={oracle},target=/app",
            "--mount",
            f"type=bind,source={task / 'solution'},target=/solution,readonly",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            image,
            "timeout",
            "--kill-after=5",
            "330",
            "bash",
            "/solution/solve.sh",
        ],
        capture_output=True,
        text=True,
        timeout=360,
        check=True,
    )
    (output / "oracle_stdout.txt").write_text(completed.stdout)
    positive = grade(image, task, oracle, output / "positive_control")
    negative_workspace = output / "negative"
    shutil.copytree(public, negative_workspace)
    negative = grade(image, task, negative_workspace, output / "negative_control")
    if not positive["evaluation"]["feasible"] or negative["evaluation"]["feasible"]:
        raise ValueError("ORAgentBench oracle/no-op controls failed")
    workspace = output / "workspace"
    shutil.copytree(public, workspace)
    (workspace / "submissions").chmod(0o777)
    response = solve(
        policy,
        (task / "instruction.md").read_text()
        + "\n/app and /workspace contain the same task files.",
        output,
        image,
    )
    pristine = output / "evaluation"
    shutil.copytree(public, pristine)
    for name in ("solution.csv", "solution.json", "model.md", "solve.py", "solve_log.md"):
        path = workspace / "submissions" / name
        if path.is_file():
            if path.is_symlink() or not path.resolve().is_relative_to(workspace.resolve()):
                raise ValueError("Invalid OR submission artifact path")
            shutil.copyfile(path, pristine / "submissions" / name)
    scored = grade(image, task, pristine, output / "grade")
    return result(
        response,
        scored["reward"]["scalar_reward"],
        scored,
        task.name,
        "Official oracle and grader; local 2CPU/4GB budget variant",
        from_scratch_replay_validated=False,
    )


def tau(policy: dict, output: Path, image: str):
    policy_path = output / "policy.json"
    write_json(policy_path, policy)
    with (output / "stdout.txt").open("w") as stdout, (output / "stderr.txt").open("w") as stderr:
        completed = subprocess.run(
            [
                str(ROOT / "data/external/tau3_bench/.venv/bin/python"),
                "-m",
                "moevo.codex.tau_pilot",
                "--output",
                str(output),
                "--count",
                "1",
                "--seed",
                str(policy["seed"]),
                "--timeout",
                str(policy["timeout_seconds"]),
                "--policy-file",
                str(policy_path),
            ],
            cwd=ROOT,
            env=account_environment(),
            stdout=stdout,
            stderr=stderr,
            timeout=policy["timeout_seconds"] + 180,
        )
    if completed.returncode:
        raise RuntimeError("Tau simulation failed: " + (output / "stderr.txt").read_text()[-1800:])
    reports = list((output / "tau3_retail").glob("task-*/report.json"))
    if len(reports) != 1:
        raise ValueError("Tau did not return exactly one task result")
    report = json.loads(reports[0].read_text())
    return {
        "score": report["reward"]["reward"],
        "grade": report["reward"],
        "task_id": report["task_id"],
        "e2e_validated": report["e2e_smoke_passed"],
        "protocol": report["protocol"],
        "termination": report["termination"],
        "model_calls": report["model_calls"],
    }


def education(policy: dict, output: Path, image: str):
    root = ROOT / "data/raw/eduagentbench"
    checks = []
    for line in (root / "checksums.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = root / name.lstrip("*")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Education dataset checksum mismatch: {name}")
        checks.append(name)
    write_json(
        output / "availability.json",
        {
            "tasks": len(json.loads((root / "tasks.json").read_text())),
            "verified_files": checks,
            "released_runtime_files": [str(p.relative_to(root)) for p in root.rglob("*.py")],
            "dataset": "https://huggingface.co/datasets/eduagentbench/eduagentbench",
            "paper": "https://arxiv.org/abs/2605.14322",
        },
    )
    raise RuntimeError(
        "EduAgentBench: all 22 asset checksums pass, but the downloaded public release has no evaluator or Canvas runtime; cannot claim official E2E validation"
    )


def putnam(policy: dict, output: Path, image: str):
    from moevo.codex.formal_tasks import putnam as validate

    return validate(policy, output, image)


HANDLERS = {
    "oragentbench": operations,
    "tau3_bench": tau,
    "eduagentbench": education,
    "gdpval": gdp,
    "harvey_lab": legal,
    "terminal_bench_2": regex,
    "terminal_bench_science": cilia,
    "putnambench": putnam,
}
