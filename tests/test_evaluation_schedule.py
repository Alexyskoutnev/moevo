"""Offline protocol tests: scheduling changes which real task calls would execute."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from pathlib import Path

import pytest

from moevo import run_discovery
from moevo.controller import MoevoController
from moevo.core.config import MoevoConfig, build_arg_parser, config_from_args
from moevo.generation.evaluator import load_evaluate_fn
from moevo.generation.schedule import EvaluationBudgetError, StagedEvaluator


def manifest_for(objectives):
    tasks = [
        {"id": f"{d}/{i}", "objective": d, "benchmark": f"bench-{d}", "task_id": str(i), "seed": 42}
        for d in objectives
        for i in range(6)
    ]
    return {
        "version": 1,
        "split": "search",
        "protocol": "offline-contract-test-v1",
        "tasks": tasks,
        "confirmation_ids": [f"{d}/{i}" for d in objectives for i in range(2)],
    }


@pytest.fixture
def setup(tmp_path):
    objectives = [f"domain{i}" for i in range(11)]
    manifest = tmp_path / "search.json"
    manifest.write_text(json.dumps(manifest_for(objectives)))
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# pinned offline adapter\n")
    calls = []

    def task_fn(path, task):
        code = Path(path).read_text()
        calls.append((code, dict(task)))
        score = 0.5 if "seed" in code else 0.8 if "good" in code else 0.0
        # A child improves most domains but regresses on an axis outside some screens.
        if "regression" in code and task["objective"] == "domain10":
            score = 0.1
        return {"status": "scored", "score": score, "feedback": "Search trace"}

    def make(fn=task_fn, **kwargs):
        return StagedEvaluator(
            fn,
            manifest_path=str(manifest),
            evaluator_path=str(evaluator),
            objectives=objectives,
            model="codex/gpt-6-astra",
            output_dir=tmp_path / "run",
            **kwargs,
        )

    return make, calls, manifest, evaluator, objectives


def test_domains_and_tasks_rotate_without_starvation(setup):
    make, _, _, _, objectives = setup
    scheduler = make()
    seen = set()
    for i in range(math.ceil(len(objectives) / 3)):
        ids = scheduler.screen_ids(i)
        assert len(ids) == len(set(ids)) == 6
        domains = [scheduler.tasks[t]["objective"] for t in ids]
        assert all(domains.count(domain) == 2 for domain in set(domains))
        seen.update(domains)
    assert seen == set(objectives)
    assert {t for i in range(11) for t in scheduler.screen_ids(i)} == set(scheduler.tasks)


async def test_pairing_cache_and_complete_confirmation(setup):
    make, calls, _, _, objectives = setup
    scheduler = make()
    parent = await scheduler.confirm("seed")
    assert scheduler.task_evaluations == 22
    ids = scheduler.screen_ids(0)
    result = await scheduler.candidate("good regression", "seed", parent.metrics)
    assert result is not None
    assert set(result.metrics) == set(objectives)
    assert result.metrics["domain10"] == 0.1  # Never inherit the parent's untested score.
    for task_id in ids:
        assert any(code == "seed" and task["id"] == task_id for code, task in calls)
        assert any(code == "good regression" and task["id"] == task_id for code, task in calls)
    assert len(calls) == len({(code, task["id"]) for code, task in calls})
    assert all(task["judge_model"] == "gpt-5.6-terra" for _, task in calls)
    before = scheduler.task_evaluations
    assert (await scheduler.confirm("good regression")).metrics == result.metrics
    assert scheduler.task_evaluations == before


async def test_rejected_candidate_never_gets_complete_fitness_and_audits_detect_misses(setup):
    make, calls, _, _, objectives = setup
    scheduler = make(audit_every=2)
    parent = await scheduler.confirm("seed")
    assert await scheduler.candidate("bad", "seed", parent.metrics) is None
    assert len([c for c, _ in calls if c == "bad"]) == 6
    assert scheduler.seen("bad")

    # A benefit exists only in a domain outside the second screen.
    screened = {scheduler.tasks[t]["objective"] for t in scheduler.screen_ids(1)}
    hidden = next(d for d in objectives if d not in screened)
    original = scheduler.evaluate_task

    def task_fn(path, task):
        if Path(path).read_text() == "hidden-benefit":
            return {"status": "scored", "score": 0.8 if task["objective"] == hidden else 0.5}
        return original(path, task)

    scheduler.evaluate_task = task_fn
    result = await scheduler.candidate("hidden-benefit", "seed", parent.metrics)
    assert result is not None and result.metrics[hidden] == 0.8
    assert scheduler.state["events"][-1]["screen_missed_improvement"]


async def test_budget_is_reserved_for_paired_stage_and_persists_on_resume(setup):
    make, calls, _, _, _ = setup
    scheduler = make(max_task_evaluations=22)
    parent = await scheduler.confirm("seed")
    with pytest.raises(EvaluationBudgetError):
        await scheduler.candidate("good", "seed", parent.metrics)
    assert len(calls) == 22  # No partial paired stage started.
    resumed = make(max_task_evaluations=100, fresh_start=False)
    assert resumed.task_evaluations == 22
    assert (await resumed.confirm("seed")).metrics == parent.metrics
    assert len(calls) == 22
    assert await resumed.candidate("good", "seed", parent.metrics) is not None


@pytest.mark.parametrize(
    "result",
    [
        {"status": "blocked", "score": 0},
        {"status": "needs_audit", "score": 0.6},
        {"status": "scored", "score": None},
        {"status": "scored", "score": True},
        {"status": "scored", "score": float("nan")},
        {"status": "scored", "score": 54},
    ],
)
async def test_invalid_results_are_charged_but_not_cached_as_model_failures(setup, result):
    make, _, _, _, _ = setup
    scheduler = make(lambda path, task: result)
    with pytest.raises(ValueError, match="valid scored result"):
        await scheduler.confirm("seed")
    assert scheduler.task_evaluations == 1
    assert not scheduler.state["cache"]
    resumed = make(lambda path, task: result, fresh_start=False)
    assert resumed.task_evaluations == 1


async def test_async_adapter_and_actual_zero_scores_are_supported(setup):
    make, _, _, _, _ = setup

    async def task_fn(path, task):
        await asyncio.sleep(0)
        assert Path(path).read_text() == "zero"
        return {"status": "scored", "score": 0, "native_metrics": {"success": False}}

    scheduler = make(task_fn)
    assert all(v == 0 for v in (await scheduler.confirm("zero")).metrics.values())


async def test_parallel_stage_finishes_inflight_work_and_stops_on_error(setup):
    make, _, _, _, _ = setup

    async def task_fn(path, task):
        if task["id"] == "domain0/0":
            raise RuntimeError("Synthetic infrastructure failure")
        await asyncio.sleep(0)
        return {"status": "scored", "score": 0.5}

    scheduler = make(task_fn, concurrency=2)
    with pytest.raises(RuntimeError, match="Synthetic"):
        await scheduler.confirm("seed")
    assert scheduler.task_evaluations == 2
    assert len(scheduler.state["cache"]) == 1


@pytest.mark.parametrize("change", ["split", "missing_axis", "missing_benchmark", "duplicate"])
def test_invalid_manifest_fails_before_calls(setup, change):
    make, calls, manifest, _, _ = setup
    data = json.loads(manifest.read_text())
    if change == "split":
        data["split"] = "test"
    elif change == "missing_axis":
        data["confirmation_ids"] = [
            t for t in data["confirmation_ids"] if not t.startswith("domain0/")
        ]
    elif change == "missing_benchmark":
        data["tasks"][5]["benchmark"] = "second-benchmark"
    else:
        data["tasks"].append(data["tasks"][0])
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        make()
    assert calls == []


@pytest.mark.parametrize("change", ["model_source", "manifest", "schedule", "judge"])
def test_protocol_change_rejects_stale_cache(setup, change):
    make, _, manifest, evaluator, _ = setup
    make()
    kwargs = {}
    if change == "model_source":
        evaluator.write_text("# updated adapter\n")
    elif change == "manifest":
        data = json.loads(manifest.read_text())
        data["protocol"] = "changed-model-or-grader-or-image"
        manifest.write_text(json.dumps(data))
    elif change == "schedule":
        kwargs["screen_tasks"] = 3
    else:
        kwargs["judge_model"] = "gpt-5.6-luna"
    with pytest.raises(ValueError, match="protocol changed"):
        make(fresh_start=False, **kwargs)


TASK_ADAPTER = """
from pathlib import Path

def evaluate_task(program_path, task):
    code = Path(program_path).read_text()
    score = 0.5 if "seed" in code else 0.8 if "good" in code else 0.0
    if "regression" in code and task["objective"] == "domain10":
        score = 0.1
    return {"status": "scored", "score": score}
"""


async def test_controller_filters_then_admits_full_vectors_and_resumes(
    setup, monkeypatch, tmp_path
):
    _, _, manifest, evaluator, objectives = setup
    evaluator.write_text(TASK_ADAPTER)
    seed = tmp_path / "seed.py"
    seed.write_text("# seed\n")
    responses = iter(["```python\n# bad\n```", "```python\n# good regression\n```"])

    async def generate(**kwargs):
        assert kwargs["model"] == "codex/gpt-6-astra"
        assert "Next paired search screen" in kwargs["user"]
        return next(responses)

    monkeypatch.setattr("moevo.controller.generate", generate)
    cfg = MoevoConfig(
        initial_program=str(seed),
        evaluator_path=str(evaluator),
        objectives=objectives,
        output_dir=str(tmp_path / "controller"),
        evaluation_manifest=str(manifest),
        iterations=2,
        diff_mode=False,
        retry_attempts=0,
    )
    result = await MoevoController(cfg).run()
    assert result.iterations_completed == 2
    assert len(result.all_programs) == 3  # Two island seeds plus one confirmed child.
    assert all("bad" not in p.solution for p in result.all_programs)
    child = next(p for p in result.all_programs if "good" in p.solution)
    assert set(child.metrics) == set(objectives)
    assert child.metrics["domain10"] == 0.1
    assert child.metadata["evaluation_signature"]
    assert result.best_program is not None and "seed" in result.best_program.solution
    cfg.fresh_start = False
    resumed = await MoevoController(cfg).run()
    assert resumed.task_evaluations == result.task_evaluations
    assert [p.id for p in resumed.all_programs] == [p.id for p in result.all_programs]


async def test_controller_budget_stop_reports_actual_iterations(setup, monkeypatch, tmp_path):
    _, _, manifest, evaluator, objectives = setup
    evaluator.write_text(TASK_ADAPTER)
    seed = tmp_path / "seed.py"
    seed.write_text("# seed\n")

    async def generate(**kwargs):
        return "```python\n# good\n```"

    monkeypatch.setattr("moevo.controller.generate", generate)
    cfg = MoevoConfig(
        initial_program=str(seed),
        evaluator_path=str(evaluator),
        objectives=objectives,
        output_dir=str(tmp_path / "controller"),
        evaluation_manifest=str(manifest),
        iterations=10,
        diff_mode=False,
        max_task_evaluations=22,
    )
    result = await MoevoController(cfg).run()
    assert result.stop_reason == "task_evaluation_budget"
    assert result.iterations_completed == 0 and result.task_evaluations == 22
    assert len(result.all_programs) == 2
    cfg.fresh_start = False
    cfg.iterations = 1
    cfg.max_task_evaluations = 100
    resumed = await MoevoController(cfg).run()
    assert resumed.iterations_completed == 1
    assert len(resumed.all_programs) == 3


def test_opt_in_protocol_and_account_default(tmp_path):
    evaluator = tmp_path / "legacy.py"
    evaluator.write_text("def evaluate(path): return {'score': 1}\n")
    with pytest.raises(AttributeError, match="evaluate_task"):
        load_evaluate_fn(str(evaluator), "evaluate_task")
    config = config_from_args(
        build_arg_parser().parse_args(
            [
                "seed.py",
                "eval.py",
                "--evaluation-manifest",
                "search.json",
                "--resume",
                "--screen-domains",
                "4",
                "--max-task-evaluations",
                "900",
            ]
        )
    )
    assert config.screen_domains == 4 and config.max_task_evaluations == 900
    assert not config.fresh_start
    assert inspect.signature(run_discovery).parameters["model"].default == "codex/gpt-6-astra"
