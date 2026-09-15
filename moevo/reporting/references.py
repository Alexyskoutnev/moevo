"""Display independently run reference arms only on their exact source panel."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

from .progress import controller_liveness

if TYPE_CHECKING:
    from pathlib import Path


def reference_comparison(
    manifest: dict, policy: dict, roots: list[Path], *, observe_process: bool = False
) -> dict:
    labels = {
        "codex_cli_task_only": "Codex CLI · no MOEvo instructions",
        "seed_reference": "Starting harness · matched rerun",
    }
    arms = {
        key: {
            "label": label,
            "status": "not_started",
            "scored": 0,
            "total": len(manifest.get("tasks", [])),
            "benchmarks": {},
        }
        for key, label in labels.items()
    }
    identity_keys = (
        "id",
        "benchmark",
        "task_id",
        "objective",
        "seed",
        "content_sha256",
        "input_assets_sha256",
    )
    expected = {t["id"]: tuple(t.get(k) for k in identity_keys) for t in manifest.get("tasks", [])}
    issues = []
    for root in roots:
        try:
            run = json.loads((root / "run.json").read_text())
            state = json.loads((root / "reference_state.json").read_text())
        except (OSError, ValueError):
            continue
        arm = run.get("baseline_id")
        if arm not in arms:
            issues.append(f"{root.name}: unrecognized baseline identity")
            continue
        observed = {t["id"]: tuple(t.get(k) for k in identity_keys) for t in run.get("tasks", [])}
        role_keys = (
            "model",
            "reasoning_effort",
            "judge_model",
            "judge_reasoning_effort",
            "authentication",
        )
        if observed != expected or any(
            run.get("policy", {}).get(k) != policy.get(k) for k in role_keys
        ):
            issues.append(f"{root.name}: task panel or model settings do not match this run")
            continue
        if arms[arm]["status"] != "not_started":
            issues.append(f"{root.name}: duplicate arm; no automatic best-run selection")
            continue
        rows = state.get("tasks", {})
        scores, benchmarks = {}, {}
        for benchmark in dict.fromkeys(t["benchmark"] for t in manifest.get("tasks", [])):
            tasks = [t for t in manifest["tasks"] if t["benchmark"] == benchmark]
            statuses = []
            for task in tasks:
                row = rows.get(task["id"], {})
                status, value = row.get("status", "pending"), row.get("score")
                statuses.append(status)
                if (
                    status == "scored"
                    and type(value) in (int, float)
                    and math.isfinite(value)
                    and 0 <= value <= 1
                ):
                    scores[task["id"]] = float(value)
            values = [scores[t["id"]] for t in tasks if t["id"] in scores]
            status = (
                "scored"
                if len(values) == len(tasks)
                else "running"
                if "running" in statuses
                else "error"
                if "error" in statuses or "interrupted" in statuses
                else "pending"
            )
            benchmarks[benchmark] = {
                "score": sum(values) / len(values) if len(values) == len(tasks) else None,
                "status": status,
                "scored": len(values),
                "total": len(tasks),
            }
        arms[arm] = {
            "label": labels[arm],
            "run_name": root.name,
            "status": state.get("status", run.get("status", "unknown")),
            "phase": state.get("phase"),
            "scored": len(scores),
            "total": len(expected),
            "attempts": state.get("task_attempts", 0),
            "benchmarks": benchmarks,
        }
        if observe_process and arms[arm]["status"] == "running":
            process = controller_liveness(root)
            arms[arm]["process"] = process
            if process["state"] == "stopped":
                arms[arm]["status"] = "controller_stopped"
                for row in benchmarks.values():
                    if row["status"] == "running":
                        row["status"] = "interrupted"
    return {
        "arms": arms,
        "issues": issues,
        "bare_astra": {
            "status": "unavailable",
            "reason": "Account access runs Astra through Codex CLI. A separate unwrapped-model score has not been measured.",
        },
    }
