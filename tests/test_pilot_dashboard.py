"""Diagnostic pilot boundaries and honest dashboard score aggregation."""

import hashlib
import json

import pytest

from experiments.evolution_dashboard import snapshot, task_outcome
from experiments.pilot_task_adapter import instructions_from_code
from experiments.verify_pilot_epoch import verify
from moevo.reporting.aggregate import aggregate_panel
from moevo.reporting.results import export_tables


@pytest.mark.parametrize(
    "code",
    [
        "import os\nINSTRUCTIONS = 'test'",
        "INSTRUCTIONS = str(42)",
        "INSTRUCTIONS = 'one'\nINSTRUCTIONS = 'two'",
        "INSTRUCTIONS = ''",
        "MODEL = 'smaller'\nINSTRUCTIONS = 'test'",
    ],
)
def test_pilot_does_not_execute_arbitrary_generated_code(code):
    with pytest.raises(ValueError):
        instructions_from_code(code)


def test_shared_instruction_literal_is_valid():
    assert (
        instructions_from_code('"""Module notes"""\nINSTRUCTIONS = "Check results."')
        == "Check results."
    )


def test_dashboard_uses_complete_vectors_and_deduplicates_island_seeds(tmp_path):
    seed = {
        "id": "seed-first",
        "solution": "PRIVATE SOURCE",
        "parent_id": None,
        "iteration": 0,
        "metrics": {"a": 0.5, "b": 0.7},
    }
    other_seed = {**seed, "id": "seed-second"}
    child = {
        "id": "child",
        "solution": "OTHER PRIVATE SOURCE",
        "parent_id": "seed-first",
        "iteration": 0,
        "metrics": {"a": 0.8, "b": 0.2},
    }
    partial = {**child, "id": "partial", "solution": "partial source", "metrics": {"a": 0.9}}
    (tmp_path / "checkpoint_0000.json").write_text(
        json.dumps(
            {
                "database": {
                    "objectives": ["a", "b"],
                    "all_programs": [seed, other_seed, child, partial],
                    "islands": [[other_seed, child]],
                }
            }
        )
    )
    data = snapshot(tmp_path)
    assert len(data["history"]) == 2
    assert data["history"][1]["step"] == 1
    assert data["champion"]["metrics"] == seed["metrics"]
    assert "PRIVATE SOURCE" not in json.dumps(data)


def test_dashboard_has_no_fabricated_zero_baseline(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"config": {"objectives": ["a", "b"]}}))
    data = snapshot(tmp_path)
    assert data["history"] == [] and data["baseline"] is None and data["champion"] is None


def test_single_task_success_is_not_displayed_as_benchmark_accuracy():
    assert task_outcome("finqa", {"score": 1}) == ("1 passed / 1 tested", "")
    assert task_outcome("finqa", {"score": 0}) == ("0 passed / 1 tested", "")
    outcome, detail = task_outcome(
        "harvey_lab",
        {
            "score": 54 / 59,
            "native_metrics": {"grade": {"all_pass": False, "n_passed": 54, "n_criteria": 59}},
        },
    )
    assert outcome == "0 passed / 1 tested" and detail == "54 of 59 checks met"


def test_aggregate_weights_domains_equally_and_compares_the_same_tasks():
    tasks = [
        {"id": "f1", "objective": "finance", "benchmark": "finqa", "task_id": "a"},
        {"id": "f2", "objective": "finance", "benchmark": "finqa", "task_id": "b"},
        {"id": "l1", "objective": "legal", "benchmark": "harvey_lab", "task_id": "c"},
    ]
    cache = {}
    for candidate, scores in [("base", [1, 0, 0.5]), ("child", [1, 1, 0.25])]:
        for task, score in zip(tasks, scores, strict=True):
            cache[candidate + task["id"]] = {
                "candidate_sha256": candidate,
                "task_id": task["id"],
                "score": score,
                "native_metrics": {"score": 0 if task["objective"] == "legal" else score},
            }
    result = aggregate_panel(tasks, cache, "base", "child")
    assert result["overall"]["baseline"] == 0.5
    assert result["overall"]["candidate"] == 0.625
    assert result["overall"]["delta_pp"] == 12.5
    assert result["benchmarks"][0]["n_tasks"] == 2
    assert result["benchmarks"][1]["candidate_native"] == 0
    # A partial new candidate is not assigned a score from fewer/easier tasks.
    del cache["childl1"]
    result = aggregate_panel(tasks, cache, "base", "child")
    assert result["overall"]["candidate"] is None
    assert result["overall"]["delta_pp"] is None
    assert result["benchmarks"][1]["candidate_evaluated"] == 0


def test_cache_replicates_do_not_inflate_independent_task_count():
    tasks = [
        {"id": f"rep-{i}", "objective": "a", "benchmark": "b", "task_id": "same-task"}
        for i in range(2)
    ]
    aggregate = aggregate_panel(tasks, {}, None, None)
    assert aggregate["overall"]["n_tasks"] == 1
    assert aggregate["overall"]["n_evaluations"] == 2


def test_export_marks_diagnostic_scope_and_leaves_unscored_values_blank(tmp_path):
    data = snapshot(tmp_path)
    export_tables(data, tmp_path / "export")
    result = json.loads((tmp_path / "export/results.json").read_text())
    assert result["independent_heldout_results"] is False
    assert result["scores"][0]["baseline_percent"] is None
    assert "not held-out" in (tmp_path / "export/results.tex").read_text()


def test_epoch_validator_requires_real_complete_child_evidence(tmp_path):
    tasks = [
        {"id": name, "benchmark": name, "task_id": f"published-{name}", "objective": name}
        for name in ("a", "b")
    ]
    (tmp_path / "search.json").write_text(
        json.dumps(
            {
                "split": "search",
                "tasks": tasks,
                "confirmation_ids": ["a", "b"],
            }
        )
    )
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "config": {
                    "max_task_evaluations": 4,
                    "objectives": ["a", "b"],
                    "model": "codex/gpt-6-astra",
                },
                "policy": {
                    "model": "gpt-6-astra",
                    "reasoning_effort": "xhigh",
                    "judge_model": "gpt-5.6-terra",
                    "judge_reasoning_effort": "medium",
                    "authentication": "codex_chatgpt_account",
                },
            }
        )
    )
    programs, cache = [], {}
    for index, (code, scores) in enumerate([("seed", [0.5, 0.5]), ("child", [0.6, 0.4])]):
        source = f"INSTRUCTIONS = {code!r}\n"
        code_hash = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        programs.append(
            {
                "id": code,
                "solution": source,
                "metrics": dict(zip(["a", "b"], scores, strict=True)),
                "iteration": 0,
                "parent_id": "seed" if index else None,
            }
        )
        for task, score in zip(tasks, scores, strict=True):
            artifact = f"tasks/{code}/{task['id']}"
            directory = tmp_path / artifact
            directory.mkdir(parents=True)
            (directory / "report.json").write_text(
                json.dumps({"e2e_validated": True, "task_id": task["task_id"], "score": score})
            )
            cache[f"{code}-{task['id']}"] = {
                "candidate_sha256": code_hash,
                "task_id": task["id"],
                "score": score,
                "native_metrics": {"score": score},
                "usage": {"artifact_path": artifact},
            }
    (tmp_path / "seed.py").write_text("INSTRUCTIONS = 'seed'\n")
    (tmp_path / "best_program.py").write_text("INSTRUCTIONS = 'seed'\n")
    (tmp_path / "checkpoint_0000.json").write_text(
        json.dumps(
            {
                "database": {
                    "objectives": ["a", "b"],
                    "all_programs": programs,
                    "islands": [programs],
                },
            }
        )
    )
    state = {
        "cache": cache,
        "task_evaluations": 4,
        "events": [{"stage": "screen", "confirmed": True}],
    }
    ledger = tmp_path / "evaluation_state.json"
    ledger.write_text(json.dumps(state))
    assert verify(tmp_path)["status"] == "passed"
    cache["child-b"]["score"] = 0.9
    ledger.write_text(json.dumps(state))
    assert verify(tmp_path)["score_mismatches"] == ["b"]
    cache["child-b"]["score"] = 0.4
    del cache["child-b"]
    ledger.write_text(json.dumps(state))
    result = verify(tmp_path)
    assert result["status"] == "incomplete_or_failed"
    assert result["vector_errors"] == ["child"]
    assert not result["checks"]["at_least_one_child_fully_evaluated"]
