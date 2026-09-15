"""Observe a real run, optionally log offline W&B metrics, and verify completion."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.evolution_dashboard import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--wandb", choices=["disabled", "offline"], default="disabled")
    args = parser.parse_args()
    root = args.run.resolve()
    tracker = None
    if args.wandb == "offline":
        import wandb

        tracker = wandb.init(
            project="moevo-development",
            name=root.name,
            dir=str(root),
            mode="offline",
            config={
                "run": root.name,
                "scope": "development integration",
                "authentication": "Codex ChatGPT account",
            },
            settings=wandb.Settings(disable_git=True, save_code=False, console="off"),
        )
    previous = None
    try:
        while True:
            data = snapshot(root)
            metrics = {
                "task_attempts": data["task_evaluations"],
                "tasks_scored": sum(b["seed_score"] is not None for b in data["benchmarks"]),
                "screens_started": data["screens_started"],
                "screens_finished": len(data["screens"]),
                "complete_evolved_versions": data["complete_evolved_versions"],
                "task_errors": data["task_errors"],
            }
            for label in ("baseline", "champion"):
                if data[label]:
                    metrics[f"{label}/aggregate_percent"] = 100 * data[label]["mean"]
                    metrics[f"{label}/weakest_domain_percent"] = 100 * data[label]["minimum"]
                    metrics.update(
                        {f"{label}/{k}_percent": 100 * v for k, v in data[label]["metrics"].items()}
                    )
            state = json.dumps({"status": data["status"], **metrics}, sort_keys=True)
            if state != previous:
                print(state, flush=True)
                if tracker:
                    tracker.log(metrics)
                previous = state
            if data["status"] in {"completed", "failed", "interrupted"}:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.evolution_dashboard",
                        "--run",
                        str(root),
                        "--export-results",
                        str(root / "export"),
                    ],
                    check=True,
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.verify_pilot_epoch",
                        "--run",
                        str(root),
                        "--output",
                        str(root / "validation.json"),
                        "--require-complete",
                    ]
                )
                raise SystemExit(result.returncode)
            time.sleep(5)
    finally:
        if tracker:
            tracker.finish()


if __name__ == "__main__":
    main()
