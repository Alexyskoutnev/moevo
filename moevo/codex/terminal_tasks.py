"""One official terminal task and one science task with isolated native tests."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from moevo.codex.container_runtime import solve_in_container
from moevo.codex.domain_tasks import ROOT, result, solve
from moevo.codex.finance_pilot import write_json


def verifier(image: str, workspace: Path, task: Path, output: Path, science: bool):
    output.mkdir(parents=True, exist_ok=True)
    mounts = ["--mount", f"type=bind,source={output},target=/logs/verifier"]
    if science:
        artifacts = workspace / "results"
        artifacts.mkdir(exist_ok=True)
        mounts += ["--mount", f"type=bind,source={artifacts},target=/root/results,readonly"]
        command = ["bash", "/tests/test.sh"]
    else:
        mounts += [
            "--mount",
            f"type=bind,source={workspace},target=/app,readonly",
            "--mount",
            f"type=bind,source={task / 'tests'},target=/tests,readonly",
        ]
        command = [
            "python",
            "-m",
            "pytest",
            "--ctrf",
            "/logs/verifier/ctrf.json",
            "/tests/test_outputs.py",
            "-rA",
        ]
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
            *mounts,
            image,
            *command,
        ],
        capture_output=True,
        text=True,
        timeout=360,
    )
    (output / "stdout.txt").write_text(completed.stdout)
    (output / "stderr.txt").write_text(completed.stderr)
    # A failed assertion is a valid zero. Collection/import/runtime failure is not.
    ctrf = json.loads((output / "ctrf.json").read_text())
    summary = ctrf["results"]["summary"]
    if summary["tests"] <= 0 or any(
        t.get("status") not in {"passed", "failed", "skipped"} for t in ctrf["results"]["tests"]
    ):
        raise ValueError("Verifier did not execute its test suite")
    if completed.returncode not in {0, 1}:
        raise RuntimeError(f"Verifier infrastructure exit {completed.returncode}")
    score = float(summary["failed"] == 0 and summary["passed"] > 0)
    grade = {"score": score, "tests": summary, "exit_code": completed.returncode}
    write_json(output / "grade.json", grade)
    return grade


def run_oracle(image: str, task: Path, workspace: Path, output: Path, science: bool):
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
            f"type=bind,source={workspace},target={'/root' if science else '/app'}",
            "--mount",
            f"type=bind,source={task / 'solution'},target=/solution,readonly",
            image,
            "timeout",
            "--kill-after=5",
            "900",
            "bash",
            "/solution/solve.sh",
        ],
        capture_output=True,
        text=True,
        timeout=930,
    )
    (output / "oracle_stdout.txt").write_text(completed.stdout)
    (output / "oracle_stderr.txt").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"Official oracle failed with exit {completed.returncode}")


def regex(policy: dict, output: Path, image: str):
    task = ROOT / "data/external/terminal_bench_2/regex-log"
    grader_image = "docker.io/library/moevo-terminal-verifier:20260915"
    oracle = output / "oracle"
    oracle.mkdir()
    run_oracle(image, task, oracle, output, False)
    positive = verifier(grader_image, oracle, task, output / "positive_control", False)
    negative_workspace = output / "negative"
    negative_workspace.mkdir()
    negative = verifier(grader_image, negative_workspace, task, output / "negative_control", False)
    if positive["score"] != 1 or negative["score"] != 0:
        raise ValueError("Terminal-Bench oracle/no-op controls failed")
    response = solve(policy, (task / "instruction.md").read_text(), output, image)
    # Only the requested artifact enters the verifier, with original tests.
    clean = output / "evaluation"
    clean.mkdir()
    artifact = output / "workspace/regex.txt"
    if artifact.is_file():
        if artifact.is_symlink() or not artifact.resolve().is_relative_to(
            (output / "workspace").resolve()
        ):
            raise ValueError("Invalid terminal artifact path")
        shutil.copyfile(artifact, clean / "regex.txt")
    grade = verifier(grader_image, clean, task, output / "grade", False)
    return result(
        response,
        grade["score"],
        grade,
        "regex-log",
        "Official Terminal-Bench 2 test code/Python 3.13; dependencies preinstalled; local agent runtime and resource variant",
    )


def cilia(policy: dict, output: Path, image: str):
    task = (
        ROOT / "data/external/terminal_bench_science/tasks/life-sciences/biology/cilia-segmentation"
    )
    agent_image = "docker.io/library/moevo-cilia-agent:20260915"
    grader_image = "docker.io/library/moevo-cilia-verifier:20260915"
    oracle = output / "oracle"
    shutil.copytree(task / "environment/data", oracle / "data")
    run_oracle(agent_image, task, oracle, output, True)
    positive = verifier(grader_image, oracle, task, output / "positive_control", True)
    negative_workspace = output / "negative"
    negative_workspace.mkdir()
    negative = verifier(grader_image, negative_workspace, task, output / "negative_control", True)
    if positive["score"] != 1 or negative["score"] != 0:
        raise ValueError("Science oracle/no-op controls failed")
    workspace = output / "workspace"
    shutil.copytree(task / "environment/data", workspace / "data")
    response = solve_in_container(
        policy["instructions"]
        + "\n\n"
        + (task / "instruction.md").read_text()
        + f"\nThis integration run has a {policy['timeout_seconds']}-second agent budget.",
        workspace,
        output / "agent",
        image=agent_image,
        timeout=policy["timeout_seconds"],
        max_tools=policy["max_tool_calls"],
        root_workspace=True,
    )
    clean = output / "evaluation"
    (clean / "results").mkdir(parents=True)
    names = [
        "well_summary.csv",
        "cilia_measurements.csv",
        *[f"cilia_image_well{i}_nuclei_mask.tif" for i in (1, 2, 3)],
    ]
    for name in names:
        artifact = workspace / "results" / name
        if artifact.is_file():
            if artifact.is_symlink() or not artifact.resolve().is_relative_to(workspace.resolve()):
                raise ValueError("Invalid science artifact path")
            shutil.copyfile(artifact, clean / "results" / name)
    grade = verifier(grader_image, clean, task, output / "grade", True)
    return result(
        response,
        grade["score"],
        grade,
        "cilia-segmentation",
        "Official science environment, oracle and separate verifier; 1200s/2CPU/4GB offline mini-run variant; source commit pinned",
    )
