"""Offline lifecycle tests for rescoring, monitor isolation, and resumable slices."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.instruction_search import (
    build_instruction_prompt,
    parse_candidate_response,
    validate_instruction_change,
)
from experiments.run_sliced_study import (
    PanelLedger,
    SlicedStudy,
    digest,
    readiness,
)
from moevo.codex.finance_pilot import write_json
from moevo.core.types import Program
from moevo.generation.schedule import EvaluationBudgetError

BENCHMARKS = ["bench-a", "bench-b", "bench-c"]


def make_study(tmp_path):
    study = tmp_path / "study"
    policy = {
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "authentication": "codex_chatgpt_account",
        "judge_model": "gpt-5.6-terra",
        "judge_reasoning_effort": "medium",
        "instructions": "original instructions",
        "timeout_seconds": 100,
        "max_tool_calls": 5,
    }
    panels = {}
    for name in [*[f"S{i}" for i in range(1, 9)], "selection", "final"]:
        tasks = [
            {
                "id": f"{name}/{benchmark}",
                "benchmark": benchmark,
                "task_id": f"task-{name}-{benchmark}",
                "objective": f"domain-{benchmark}",
                "seed": 99,
                "component": f"group-{name}-{benchmark}",
                "content_sha256": digest([name, benchmark]),
                "adapter_ready": True,
            }
            for benchmark in BENCHMARKS
        ]
        panels[name] = {
            "version": 1,
            "split": "search" if name.startswith("S") else name,
            "slice": name,
            "protocol": "test-protocol",
            "tasks": tasks,
            "confirmation_ids": [task["id"] for task in tasks],
        }
    allocation = {"components": [name for name in panels], "final_evaluation_allowed": False}
    allocation["manifest_sha256"] = hashlib.sha256(
        json.dumps(allocation, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "benchmarks": BENCHMARKS,
        "protocol": {"policy": policy, "allocation_sha256": allocation["manifest_sha256"]},
        "missing_benchmarks": {},
        "unsupported_tasks": [],
        "task_attempt_caps": {"total": 168},
    }
    summary["study_sha256"] = digest({"summary": summary, "manifests": panels})
    write_json(study / "summary.json", summary)
    write_json(study / "allocation.json", allocation)
    for name, panel in panels.items():
        write_json(study / "panels" / f"{name}.json", panel)
    return study


def rehash(study):
    summary = json.loads((study / "summary.json").read_text())
    summary.pop("study_sha256")
    panels = {path.stem: json.loads(path.read_text()) for path in (study / "panels").glob("*.json")}
    summary["study_sha256"] = digest({"summary": summary, "manifests": panels})
    write_json(study / "summary.json", summary)


def fake_runner(tmp_path, *, fail_slice=None, same_candidate=False, identity=None):
    study = make_study(tmp_path) if not (tmp_path / "study").exists() else tmp_path / "study"
    calls, searches = [], []
    failure = [fail_slice]

    def execute(path, task):
        stage = Path(os.environ["MOEVO_SLICED_DIR"])
        config = json.loads((stage / "run.json").read_text())
        role = config["execution_role"]
        panel = json.loads(Path(config["panel_path"]).read_text())
        assert role in {"search", "selection"}
        assert task["id"] in {item["id"] for item in panel["tasks"]}
        code = Path(path).read_text()
        original = "original instructions" in code
        value = (
            0.5
            if original
            else 0.4
            if role == "selection"
            else 0.8
            if panel["slice"] in code
            else 0.2
        )
        calls.append(
            {
                "role": role,
                "slice": panel["slice"],
                "task_id": task["id"],
                "code": code,
                "score": value,
            }
        )
        return {
            "status": "scored",
            "score": value,
            "feedback": "PRIVATE_MONITOR_FEEDBACK" if role == "selection" else "search feedback",
            "native_metrics": {"score": value},
            "usage": {"tokens": 1},
        }

    async def search(config):
        stage = Path(config.output_dir)
        panel = json.loads(Path(config.evaluation_manifest).read_text())
        searches.append((panel["slice"], Path(config.initial_program).read_text()))
        assert config.retry_attempts == 0 and config.iterations == 4
        assert config.selection == "nsga3" and config.evaluation_concurrency == 2
        if panel["slice"] == failure[0]:
            failure[0] = None
            raise RuntimeError("injected search interruption")
        code = (
            Path(config.initial_program).read_text()
            if same_candidate
            else f'INSTRUCTIONS = "candidate-{panel["slice"]}"\n'
        )
        candidate = stage / "test_candidate.py"
        candidate.write_text(code)
        ledger = PanelLedger(
            stage / "evaluation_state.json",
            digest(panel),
            panel["tasks"],
            config.max_task_evaluations,
            execute,
            {},
        )
        await ledger.ensure(Path(config.initial_program))
        await ledger.ensure(candidate)
        return SimpleNamespace(
            best_program=SimpleNamespace(solution=code, id="fake-chosen"),
            iterations_completed=4,
            stop_reason="iterations_completed",
            pareto_front=[1],
            hypervolume=0.5,
        )

    runner = SlicedStudy(
        study,
        tmp_path / "run",
        17,
        "sha256:runtime",
        task_executor=execute,
        search_executor=search,
        adapter=Path(__file__),
        source_identity=identity or {"offline_executor": "v1"},
        required_benchmarks=BENCHMARKS,
    )
    return runner, calls, searches


async def test_next_slice_rescores_incoming_harness_and_compares_original_on_same_panel(tmp_path):
    runner, calls, searches = fake_runner(tmp_path)
    state = await runner.run(through_slice=2)
    assert state["status"] == "partial_complete"
    assert [name for name, _ in searches] == ["S1", "S2"]
    assert "candidate-S1" in searches[1][1]
    incoming = [
        row
        for row in calls
        if row["role"] == "search" and row["slice"] == "S2" and "candidate-S1" in row["code"]
    ]
    assert len(incoming) == 3 and all(row["score"] == 0.2 for row in incoming)
    assert state["slices"]["S2"]["incoming_sha256"] == state["slices"]["S1"]["chosen_sha256"]
    for entry in state["slices"].values():
        assert entry["development"]["overall"]["baseline"] == 0.5
        assert entry["development"]["overall"]["delta_pp"] == pytest.approx(30)
        assert entry["selection"]["overall"]["delta_pp"] == pytest.approx(-10)
    assert not any(row["slice"] == "final" for row in calls)


async def test_completed_stage_resume_reuses_identical_observations_and_monitor_seed(tmp_path):
    runner, calls, searches = fake_runner(tmp_path)
    await runner.run(through_slice=2)
    before = len(calls), len(searches)
    await runner.run(resume=True, through_slice=2)
    assert (len(calls), len(searches)) == before
    original_monitor = [
        row
        for row in calls
        if row["role"] == "selection" and "original instructions" in row["code"]
    ]
    assert len(original_monitor) == 3
    ledger = json.loads((tmp_path / "run/selection_monitor/evaluation_state.json").read_text())
    assert "PRIVATE_MONITOR_FEEDBACK" not in json.dumps(ledger)
    assert len((tmp_path / "run/metrics.jsonl").read_text().splitlines()) == 2


async def test_unchanged_candidate_reuses_current_baseline_and_fixed_monitor(tmp_path):
    runner, calls, _ = fake_runner(tmp_path, same_candidate=True)
    state = await runner.run(through_slice=2)
    assert len(calls) == 9  # Three new tasks per search slice, plus one fixed monitor.
    assert not (tmp_path / "run/slices/S2/original_seed").exists()
    assert state["slices"]["S2"]["development"]["overall"]["delta_pp"] == 0


async def test_resume_interruption_preserves_finished_slice_and_charges_failed_task(tmp_path):
    runner, calls, searches = fake_runner(tmp_path, fail_slice="S2")
    with pytest.raises(RuntimeError, match="injected"):
        await runner.run(through_slice=2)
    assert runner.state["slices"]["S1"]["complete"]
    before_s1 = len([row for row in calls if row["slice"] == "S1"])
    result = await runner.run(resume=True, through_slice=2)
    assert result["status"] == "partial_complete"
    assert len([row for row in calls if row["slice"] == "S1"]) == before_s1
    assert [name for name, _ in searches] == ["S1", "S2", "S2"]


async def test_changed_sources_or_chosen_harness_block_resume(tmp_path):
    runner, _, _ = fake_runner(tmp_path)
    await runner.run(through_slice=1)
    other, calls, _ = fake_runner(tmp_path, identity={"offline_executor": "v2"})
    with pytest.raises(ValueError, match="identity changed"):
        await other.run(resume=True, through_slice=1)
    assert not calls
    (tmp_path / "run/slices/S1/chosen.py").write_text('INSTRUCTIONS="tampered"')
    with pytest.raises(ValueError, match="chosen source changed"):
        await runner.run(resume=True, through_slice=1)


def test_readiness_rejects_missing_benchmark_and_unsupported_terminal_task(tmp_path):
    study = make_study(tmp_path)
    panel = json.loads((study / "panels/S1.json").read_text())
    panel["tasks"][0]["adapter_ready"] = False
    write_json(study / "panels/S1.json", panel)
    rehash(study)
    result = readiness(study, required_benchmarks=BENCHMARKS)
    assert not result["ready"] and any("unsupported" in item for item in result["blockers"])
    assert result["model_calls"] == 0
    assert not readiness(study)["ready"]  # Real CLI requires all thirteen benchmarks.


def test_future_blocker_can_only_be_deferred_by_explicit_stop_boundary(tmp_path):
    study = make_study(tmp_path)
    panel = json.loads((study / "panels/S8.json").read_text())
    panel["tasks"][0]["adapter_ready"] = False
    write_json(study / "panels/S8.json", panel)
    rehash(study)
    assert readiness(study, through_slice=1, required_benchmarks=BENCHMARKS)["ready"]
    assert not readiness(study, required_benchmarks=BENCHMARKS)["ready"]


async def test_full_panel_failures_are_charged_and_never_cached_as_zero(tmp_path):
    path = tmp_path / "evaluation_state.json"
    program = tmp_path / "seed.py"
    program.write_text("seed")
    calls = []

    def fail(path, task):
        calls.append(task["id"])
        assert json.loads((tmp_path / "evaluation_state.json").read_text())["task_evaluations"] >= 1
        raise RuntimeError("grader failed")

    ledger = PanelLedger(path, "identity", [{"id": "a"}], 1, fail, {})
    with pytest.raises(RuntimeError, match="grader failed"):
        await ledger.ensure(program)
    assert ledger.state["task_evaluations"] == 1 and not ledger.state["cache"]
    resumed = PanelLedger(path, "identity", [{"id": "a"}], 1, fail, {})
    with pytest.raises(EvaluationBudgetError):
        await resumed.ensure(program)
    assert calls == ["a"]


async def test_final_role_is_never_authorized(tmp_path):
    runner, calls, _ = fake_runner(tmp_path)
    with pytest.raises(ValueError, match="Final evaluation is prohibited"):
        await runner._full_panel(
            tmp_path / "final", runner.data["panels"]["final"], "selection", tmp_path / "seed.py", 3
        )
    assert not calls


async def test_proposal_is_durably_charged_before_model_execution(tmp_path, monkeypatch):
    import moevo.controller as controller

    runner, _, _ = fake_runner(tmp_path)
    await runner.run(through_slice=1)

    async def generated(**kwargs):
        saved = json.loads((tmp_path / "run/run.json").read_text())
        assert saved["proposal_calls"] == 1
        raise RuntimeError("proposal failure")

    monkeypatch.setattr(controller, "generate", generated)
    with runner._proposal_ledger(), pytest.raises(RuntimeError, match="proposal failure"):
        parent = Program(id="parent", solution=runner.original.read_text(), metrics={"score": 0.5})
        system, user = controller.build_prompt(parent, [], ["score"])
        await controller.generate(model="codex/gpt-6-astra", system=system, user=user)
    assert runner.state["proposal_calls"] == 1
    runner.state["proposal_calls"] = 32
    with runner._proposal_ledger(), pytest.raises(EvaluationBudgetError):
        await controller.generate(model="test", system="", user="")


async def test_all_eight_slices_finish_with_distinct_panels_and_fixed_monitor(tmp_path):
    runner, calls, searches = fake_runner(tmp_path)
    state = await runner.run()
    assert state["status"] == "completed" and len(state["completed_slices"]) == 8
    assert len(searches) == 8
    assert len({row["task_id"] for row in calls if row["role"] == "search"}) == 24
    assert len({row["task_id"] for row in calls if row["role"] == "selection"}) == 3
    assert state["task_attempts"] == len(calls) == 96
    latest = state["latest_development_chosen_sha256"]
    resumed = await runner.run(resume=True, through_slice=1)
    assert (
        resumed["status"] == "completed" and resumed["latest_development_chosen_sha256"] == latest
    )


async def test_real_controller_path_counts_proposals_and_never_receives_monitor_feedback(
    tmp_path, monkeypatch
):
    import moevo.controller as controller

    study = make_study(tmp_path)
    proposals = []
    roles = []

    async def generate(**kwargs):
        assert kwargs["model"] == "codex/gpt-6-astra"
        assert "PRIVATE_MONITOR_FEEDBACK" not in kwargs["user"]
        assert "PRIVATE_MONITOR_FEEDBACK" not in kwargs["system"]
        proposals.append(kwargs["user"])
        return f'```python\nINSTRUCTIONS = "revision-{len(proposals)}"\n```'

    def execute(path, task):
        stage = Path(os.environ["MOEVO_SLICED_DIR"])
        config = json.loads((stage / "run.json").read_text())
        role = config["execution_role"]
        roles.append(role)
        code = Path(path).read_text()
        version = int(code.split("revision-")[1].split('"')[0]) if "revision-" in code else 0
        value = 0.1 if role == "selection" and version else 0.2 + version * 0.02
        return {
            "status": "scored",
            "score": value,
            "feedback": "PRIVATE_MONITOR_FEEDBACK" if role == "selection" else "search-only signal",
        }

    monkeypatch.setattr(controller, "generate", generate)
    monkeypatch.setattr(controller, "load_evaluate_fn", lambda *args: execute)
    runner = SlicedStudy(
        study,
        tmp_path / "run",
        17,
        "sha256:runtime",
        task_executor=execute,
        adapter=Path(__file__),
        source_identity={"offline": "v1"},
        required_benchmarks=BENCHMARKS,
    )
    state = await runner.run(through_slice=2)
    assert len(proposals) == state["proposal_calls"] == 8
    assert state["task_attempts"] == len(roles) == 42
    assert state["slices"]["S2"]["selection"]["overall"]["delta_pp"] < 0
    assert state["slices"]["S2"]["development"]["overall"]["delta_pp"] > 0


def test_instruction_prompt_preserves_parent_context_and_search_objectives():
    parent = Program(
        "p", 'INSTRUCTIONS = "verify units"\n', {"finance": 0.4}, feedback="wrong unit"
    )
    context = Program("c", 'INSTRUCTIONS = "check constraints"\n', {"finance": 0.6})
    system, user = build_instruction_prompt(parent, [context], ["finance"], explore=True)
    assert "do not evolve executable harness code" in system
    assert "entire file is yours" not in system + user
    for value in ("verify units", "check constraints", "wrong unit", "finance", "0.4", "0.6"):
        assert value in user
    assert "diversity exploration" in user


def patch(search, replacement):
    return f"<<<<<<< SEARCH\n{search}=======\n{replacement}>>>>>>> REPLACE\n"


def test_patch_application_is_atomic_and_each_block_must_match_once():
    parent = 'INSTRUCTIONS = "one"\n'
    first = patch(parent, 'INSTRUCTIONS = "two"\n')
    second = patch('INSTRUCTIONS = "two"\n', 'INSTRUCTIONS = "three"\n')
    assert parse_candidate_response(first + second, parent) == 'INSTRUCTIONS = "three"\n'
    with pytest.raises(ValueError, match="matched 0 times"):
        parse_candidate_response(first + patch("missing\n", "new\n"), parent)
    with pytest.raises(ValueError, match="matched 2 times"):
        parse_candidate_response(patch("one\n", "two\n"), "one\none\n")
    with pytest.raises(ValueError, match="Malformed"):
        parse_candidate_response(first + "<<<<<<< SEARCH\nmissing delimiter", parent)
    assert parent == 'INSTRUCTIONS = "one"\n'


@pytest.mark.parametrize(
    "candidate",
    [
        'INSTRUCTIONS="old"\n# cosmetic only\n',
        'INSTRUCTIONS="new"\nimport os\n',
        'INSTRUCTIONS=str("new")\n',
        'INSTRUCTIONS="first"\nINSTRUCTIONS="second"\n',
        'INSTRUCTIONS=""\n',
    ],
)
def test_only_changed_literal_instructions_are_evaluable(candidate):
    with pytest.raises(ValueError):
        validate_instruction_change('INSTRUCTIONS="old"\n', candidate)


async def test_proposal_artifacts_and_actual_codex_usage_are_saved_before_evaluation(
    tmp_path, monkeypatch
):
    import moevo.codex.client as client
    import moevo.controller as controller

    runner, _, _ = fake_runner(tmp_path)
    await runner.run(through_slice=1)
    monkeypatch.setenv("MOEVO_ACCOUNT_ONLY", "1")
    original_parser = controller.parse_response
    original_builder = controller.build_prompt
    parent = Program("p", runner.original.read_text(), {"score": 0.5}, feedback="search only")

    def run_codex(prompt, **kwargs):
        assert kwargs["model"] == "gpt-6-astra" and kwargs["effort"] == "xhigh"
        assert kwargs["tools"] is False
        directory = kwargs["log_dir"].parent
        request = json.loads((directory / "request.json").read_text())
        assert request["parent"]["id"] == "p" and request["proposal_budget_charge"] == 1
        assert request["system"] in prompt and request["user"] in prompt
        assert (directory / "parent.py").read_text() == parent.solution
        assert json.loads((runner.output / "run.json").read_text())["proposal_calls"] == 1
        assert not (directory / "parsed_candidate.py").exists()
        return client.CodexResponse(
            '```python\nINSTRUCTIONS = "check assumptions and units"\n```',
            [{"type": "turn.completed", "usage": {"input_tokens": 23, "output_tokens": 7}}],
            {"input_tokens": 23, "output_tokens": 7},
            1.2,
        )

    monkeypatch.setattr(client, "run_codex", run_codex)
    before = runner.state["task_attempts"]
    with runner._proposal_ledger():
        system, user = controller.build_prompt(parent, [], ["score"])
        response = await controller.generate(model="codex/gpt-6-astra", system=system, user=user)
        code = controller.parse_response(response, parent_code=parent.solution, diff_mode=True)
        assert code is not None and "check assumptions and units" in code
    assert (
        controller.parse_response is original_parser and controller.build_prompt is original_builder
    )
    directory = runner.output / "proposals/call-0001"
    status = json.loads((directory / "status.json").read_text())
    assert status["status"] == "validated_before_evaluation"
    assert status["usage"] == {"input_tokens": 23, "output_tokens": 7}
    assert status["duration_seconds"] == 1.2
    assert (directory / "raw_response.txt").exists() and (directory / "events.jsonl").exists()
    assert "original instructions" in (directory / "candidate.diff").read_text()
    assert runner.state["task_attempts"] == before


async def test_invalid_model_proposals_never_enter_task_evaluation(tmp_path, monkeypatch):
    import moevo.controller as controller

    study = make_study(tmp_path)
    tasks = []

    def execute(path, task):
        tasks.append(task["id"])
        assert "original instructions" in Path(path).read_text()
        return {"status": "scored", "score": 0.5}

    async def invalid(**kwargs):
        return '```python\nINSTRUCTIONS = "new"\nimport os\n```'

    monkeypatch.setattr(controller, "generate", invalid)
    monkeypatch.setattr(controller, "load_evaluate_fn", lambda *args: execute)
    runner = SlicedStudy(
        study,
        tmp_path / "run",
        17,
        "runtime",
        task_executor=execute,
        adapter=Path(__file__),
        source_identity={"fake": "v1"},
        required_benchmarks=BENCHMARKS,
    )
    result = await runner.run(through_slice=1)
    assert result["proposal_calls"] == 4
    assert result["task_attempts"] == len(tasks) == 6  # Seed search + fixed monitor only.
    assert all(item["status"] == "rejected" for item in result["proposal_events"])


def test_gzipped_allocation_preserves_decoded_identity_checks(tmp_path):
    study = make_study(tmp_path)
    path = study / "allocation.json"
    (study / "allocation.json.gz").write_bytes(gzip.compress(path.read_bytes()))
    path.unlink()
    assert readiness(study, required_benchmarks=BENCHMARKS)["ready"]
    allocation = json.loads(gzip.decompress((study / "allocation.json.gz").read_bytes()))
    allocation["components"].append("tampered")
    (study / "allocation.json.gz").write_bytes(gzip.compress(json.dumps(allocation).encode()))
    assert not readiness(study, required_benchmarks=BENCHMARKS)["ready"]
