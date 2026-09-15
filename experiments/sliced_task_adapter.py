"""Exact-ID adapter for new search slices and a separate selection monitor."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path

from experiments.pilot_task_adapter import instructions_from_code
from experiments.prepare_multidomain_study import write_json


def backend(benchmark: str):
    from moevo.codex import multitask_core, multitask_extended

    return multitask_core if benchmark in multitask_core.DOMAINS else multitask_extended


def evaluate_task(program_path: str, task: dict) -> dict:
    root = Path(os.environ["MOEVO_SLICED_DIR"]).resolve()
    run = json.loads((root / "run.json").read_text())
    manifest = json.loads(Path(run["panel_path"]).read_text())
    role = run["execution_role"]
    if role not in {"search", "selection"} or manifest.get("split") != role:
        raise ValueError("Final evaluation is not authorized by this development adapter")
    expected = next((t for t in manifest["tasks"] if t["id"] == task["id"]), None)
    if expected is None or any(
        task.get(k) != expected[k] for k in ("benchmark", "task_id", "objective", "seed")
    ):
        raise ValueError("Task identity does not match the frozen panel")
    policy = {
        **run["policy"],
        "instructions": instructions_from_code(Path(program_path).read_text()),
    }
    if (
        policy.get("authentication") != "codex_chatgpt_account"
        or policy.get("model") != "gpt-6-astra"
        or policy.get("judge_model") != "gpt-5.6-terra"
    ):
        raise ValueError("This protocol requires account Astra solving and Terra rubric judging")
    module = backend(task["benchmark"])
    check = module.preflight(task["benchmark"], task["task_id"])
    if not check.get("structural_preflight_passed") or check.get("adapter_ready") is False:
        raise ValueError(f"Exact-task preflight failed: {check}")
    observed = check.get("content_sha256", check.get("input_assets_sha256"))
    if observed != expected["content_sha256"]:
        raise ValueError("Dataset content changed after panel preparation")
    if (
        expected.get("input_assets_sha256") is not None
        and check.get("input_assets_sha256") != expected["input_assets_sha256"]
    ):
        raise ValueError("Task input files changed after panel preparation")
    source = Path(program_path).read_text()
    candidate = hashlib.sha256(source.encode()).hexdigest()
    output = root / "tasks" / task["benchmark"] / f"{candidate[:12]}-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True)
    try:
        report = module.evaluate(
            task["benchmark"], task["task_id"], policy, output, run["runtime_image"]
        )
        write_json(output / "report.json", report)
        if report.get("task_id") != task["task_id"] or not report.get("e2e_validated"):
            raise ValueError("Evaluation did not validate the requested task")
        native = report["score"]
        value = (
            report.get("criterion_pass_rate", native)
            if task["benchmark"] == "harvey_lab"
            else native
        )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Task has no finite score")
        score = min(1.0, max(0.0, float(value)))
        # Detailed raw grades stay in the task artifact. The mutator receives no answer keys.
        feedback = {
            "benchmark": task["benchmark"],
            "score": score,
            "protocol": report.get("protocol"),
            "failure_summary": report.get("failure_summary"),
        }
        return {
            "status": "scored",
            "score": score,
            "native_metrics": {"score": native, "grade": report["grade"]},
            "usage": {
                "solver": report.get("usage", {}),
                "model_calls": report.get("model_calls"),
                "artifact_path": str(output.relative_to(root)),
            },
            "feedback": json.dumps(feedback),
        }
    except Exception as exc:
        write_json(output / "error.json", {"error": str(exc)})
        raise
