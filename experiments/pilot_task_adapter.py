"""Explicit single-fixture adapter for the first diagnostic evolution epoch.

This evolves the shared instruction component. It does not pretend the existing
mini handlers support arbitrary task IDs or expanded, held-out benchmark panels.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import uuid
from pathlib import Path

from experiments.run_mini_suite import handler_for
from moevo.codex.finance_pilot import write_json


def instructions_from_code(code: str) -> str:
    values = []
    for node in ast.parse(code).body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "INSTRUCTIONS"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            values.append(node.value.value)
        else:
            raise ValueError("Diagnostic pilot edits only the literal INSTRUCTIONS string")
    if len(values) != 1 or not 1 <= len(values[0].strip()) <= 8000:
        raise ValueError("Supply exactly one nonempty INSTRUCTIONS string, at most 8000 characters")
    return values[0]


def evaluate_task(program_path: str, task: dict) -> dict:
    run_root = Path(os.environ["MOEVO_PILOT_DIR"]).resolve()
    config = json.loads((run_root / "run.json").read_text())
    manifest = json.loads((run_root / "search.json").read_text())
    expected = next(t for t in manifest["tasks"] if t["id"] == task["id"])
    if any(task[k] != expected[k] for k in ("objective", "benchmark", "task_id", "seed")):
        raise ValueError("Task does not match the frozen diagnostic fixture")
    code = Path(program_path).read_text()
    policy = {
        **config["policy"],
        "instructions": instructions_from_code(code),
        "judge_model": task["judge_model"],
        "judge_reasoning_effort": task["judge_reasoning_effort"],
    }
    candidate_hash = hashlib.sha256(code.encode()).hexdigest()
    output = (
        run_root / "tasks" / task["benchmark"] / f"{candidate_hash[:12]}-{uuid.uuid4().hex[:8]}"
    )
    output.mkdir(parents=True)
    try:
        row = handler_for(task["benchmark"])(policy, output, config["runtime_image"])
        write_json(output / "report.json", row)
        if not row.get("e2e_validated") or row["task_id"] != task["task_id"]:
            raise ValueError("Adapter validity/task identity check failed")
        native = float(row["score"])
        # Fixed mappings declared before this diagnostic run. Preserve native metrics.
        score = (
            row.get("criterion_pass_rate", native) if task["benchmark"] == "harvey_lab" else native
        )
        score = max(0.0, min(1.0, score))
        feedback = {
            "benchmark": task["benchmark"],
            "native_score": native,
            "search_score": score,
            "grade": row["grade"],
        }
        return {
            "status": "scored",
            "score": score,
            "native_metrics": {"score": native, "grade": row["grade"]},
            "usage": {
                "solver": row.get("usage", {}),
                "model_calls": row.get("model_calls"),
                "artifact_path": str(output.relative_to(run_root)),
            },
            "feedback": json.dumps(feedback, ensure_ascii=False),
        }
    except Exception as exc:
        write_json(output / "error.json", {"error": str(exc)})
        raise
