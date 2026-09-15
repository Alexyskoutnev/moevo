"""Do not silently compare different tasks, incomplete scores or best-of runs."""

import json

from moevo.reporting.references import reference_comparison


def write_arm(root, tasks, rows, **overrides):
    root.mkdir()
    run = {
        "baseline_id": "codex_cli_task_only",
        "tasks": tasks,
        "policy": {"model": "astra"},
        **overrides,
    }
    (root / "run.json").write_text(json.dumps(run))
    (root / "reference_state.json").write_text(json.dumps({"status": "running", "tasks": rows}))


def test_reference_waits_for_complete_benchmark_and_keeps_zero(tmp_path):
    tasks = [
        {"id": "a", "benchmark": "b", "task_id": "one"},
        {"id": "c", "benchmark": "b", "task_id": "two"},
    ]
    root = tmp_path / "arm"
    write_arm(
        root,
        tasks,
        {"a": {"status": "scored", "score": 0}, "c": {"status": "error", "score": None}},
    )
    result = reference_comparison({"tasks": tasks}, {"model": "astra"}, [root])
    arm = result["arms"]["codex_cli_task_only"]
    assert arm["scored"] == 1 and arm["benchmarks"]["b"]["score"] is None
    assert arm["benchmarks"]["b"]["status"] == "error"
    assert result["bare_astra"]["status"] == "unavailable"


def test_reference_rejects_different_task_or_model(tmp_path):
    tasks = [{"id": "a", "benchmark": "b", "task_id": "one"}]
    for mismatch in ("task", "model"):
        root = tmp_path / mismatch
        write_arm(
            root,
            [{**tasks[0], "task_id": "other"}] if mismatch == "task" else tasks,
            {},
            policy={"model": "other" if mismatch == "model" else "astra"},
        )
        result = reference_comparison({"tasks": tasks}, {"model": "astra"}, [root])
        assert result["issues"] and result["arms"]["codex_cli_task_only"]["status"] == "not_started"


def test_reference_never_selects_best_of_duplicate_runs(tmp_path):
    tasks = [{"id": "a", "benchmark": "b", "task_id": "one"}]
    first, second = tmp_path / "first", tmp_path / "second"
    write_arm(first, tasks, {"a": {"status": "scored", "score": 0}})
    write_arm(second, tasks, {"a": {"status": "scored", "score": 1}})
    result = reference_comparison({"tasks": tasks}, {"model": "astra"}, [first, second])
    assert result["arms"]["codex_cli_task_only"]["benchmarks"]["b"]["score"] == 0
    assert "duplicate arm" in result["issues"][0]
