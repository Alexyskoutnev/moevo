"""ORAgentBench smoke validation with official oracle/grader and account Astra."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from moevo.codex.client import require_chatgpt_login
from moevo.codex.container_runtime import solve_in_container
from moevo.codex.finance_pilot import write_json


def grade(image: str, task: Path, app: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
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
            "--pids-limit=256",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=512m",
            "--mount",
            f"type=bind,source={app},target=/app,readonly",
            "--mount",
            f"type=bind,source={task / 'tests'},target=/tests,readonly",
            "--mount",
            f"type=bind,source={output},target=/logs/verifier",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            image,
            "bash",
            "/tests/test.sh",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=360,
    )
    evaluation = json.loads((output / "evaluation.json").read_text())
    details = json.loads((output / "reward_details.json").read_text())
    if details.get("quality_status") in {"no_reference_full_credit", "missing_agent_objective"}:
        raise ValueError("Official quality score lacks a valid reference/objective")
    return {"evaluation": evaluation, "reward": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="industrial_water_reuse_blending")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    root = Path("data/external/oragentbench").resolve()
    task = root / "harbor_tasks" / args.task
    output = (Path("results/domain_validation/oragentbench") / args.task).resolve()
    if (output / "report.json").exists():
        print((output / "report.json").read_text())
        return
    public = task / "environment/app"
    image = json.loads(
        subprocess.check_output(
            ["docker", "image", "inspect", "docker.io/library/moevo-or-runtime:20260915"],
            text=True,
            timeout=30,
        )
    )[0]["Id"]
    # Reference solutions and hidden graders never enter the agent workspace.
    oracle = output / "oracle_workspace"
    shutil.copytree(public, oracle, dirs_exist_ok=True)
    reference_run = subprocess.run(
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
            "--pids-limit=256",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=512m",
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
        check=True,
        timeout=360,
    )
    (output / "oracle_stdout.txt").write_text(reference_run.stdout)
    positive = grade(image, task, oracle, output / "positive_control")
    wrong = output / "negative_workspace"
    shutil.copytree(public, wrong, dirs_exist_ok=True)
    negative = grade(image, task, wrong, output / "negative_control")
    if not positive["evaluation"].get("feasible") or negative["evaluation"].get("feasible"):
        raise ValueError("Oracle/missing-solution controls failed")
    workspace = output / "workspace"
    shutil.copytree(public, workspace, dirs_exist_ok=True)
    (workspace / "submissions").chmod(0o777)
    version = require_chatgpt_login()
    response = solve_in_container(
        (task / "instruction.md").read_text()
        + "\nUse the benchmark run tool. /app and /workspace refer to the same task files.",
        workspace,
        output / "agent",
        image=image,
        timeout=args.timeout,
    )
    # Grade against pristine inputs, not any data files the agent may have edited.
    evaluated = output / "evaluation_workspace"
    shutil.copytree(public, evaluated, dirs_exist_ok=True)
    artifacts = workspace / "submissions"
    for name in ("solution.csv", "solution.json", "model.md", "solve.py", "solve_log.md"):
        path = artifacts / name
        if path.is_symlink():
            raise ValueError("Submission artifacts must be regular files")
        if path.is_file():
            shutil.copyfile(path, evaluated / "submissions" / name)
    scored = grade(image, task, evaluated, output / "grade")
    report = {
        "dataset": "ORAgentBench",
        "task_id": args.task,
        "image_id": image,
        "source": json.loads((root / "source_manifest.json").read_text()),
        "model": "gpt-6-astra",
        "effort": "xhigh",
        "cli_version": version,
        "positive_control": positive,
        "negative_control": negative,
        "grade": scored,
        "e2e_smoke_passed": True,
        "headroom_confirmed": False,
        "official_resource_budget": False,
        "protocol": "2 CPU / 4 GiB account-CLI smoke; reference base runtime; no skill bundle",
        "from_scratch_replay_validated": False,
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
