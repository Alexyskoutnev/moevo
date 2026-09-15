import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from moevo.codex import multitask_extended as extended
from moevo.codex.client import CodexResponse


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


@pytest.fixture
def dataset_root(tmp_path):
    for name in extended.DOMAINS:
        raw = name in {"gdpval", "healthbench_professional"}
        source = tmp_path / "data" / ("raw" if raw else "external") / name
        write(
            source / ("_moevo_manifest.json" if raw else "source_manifest.json"),
            {"repository": "fixture", "revision": "pinned-fixture", "files": {}},
        )
    gdp = tmp_path / "data/raw/gdpval"
    rows = []
    for index in (1, 2):
        ref = f"reference_files/record-{index}.txt"
        (gdp / ref).parent.mkdir(exist_ok=True)
        (gdp / ref).write_text(f"reference {index}")
        rows.append(
            {
                "task_id": f"gdp-{index}",
                "prompt": f"Task {index}",
                "reference_files": [ref],
                "deliverable_files": ["answer.docx"],
                "rubric_json": json.dumps(
                    [{"score": index, "criterion": f"private rubric {index}"}]
                ),
            }
        )
    (gdp / "data").mkdir()
    pq.write_table(pa.Table.from_pylist(rows), gdp / "data/data.parquet")
    legal = tmp_path / "data/external/harvey_lab/tasks/contracts/test"
    for index in (1, 2):
        task = legal / f"scenario-0{index}"
        write(
            task / "task.json",
            {
                "title": f"Legal {index}",
                "instructions": f"Legal task {index}",
                "criteria": [
                    {"id": "C1", "title": "Criterion", "match_criteria": "Correct clause"}
                ],
            },
        )
        (task / "documents").mkdir()
        (task / "documents/input.txt").write_text("document")
    health = tmp_path / "data/raw/healthbench_professional/healthbench_professional_eval.jsonl"
    tasks = [
        {
            "id": f"health-{index}",
            "conversation": {
                "messages": [
                    {"role": "user", "content": f"Medical {index}\u2028still one JSONL row"}
                ]
            },
            "rubric_items": [{"points": 1, "criterion_text": "private rubric"}],
        }
        for index in (1, 2)
    ]
    with health.open("w") as stream:
        for task in tasks:
            stream.write(json.dumps(task, ensure_ascii=False) + "\n")
    putnam = tmp_path / "data/external/putnambench/lean4/src"
    putnam.mkdir(parents=True)
    (putnam / "putnam_2024_a1.lean").write_text(
        "import Mathlib\ntheorem putnam_2024_a1 : True := sorry\n"
    )
    (putnam / "putnam_2024_a2.lean").write_text(
        "abbrev answer : Nat := sorry\ntheorem putnam_2024_a2 : True := sorry\n"
    )
    for name, task_ids in (
        ("oragentbench", ["family_1", "family_2"]),
        ("terminal_bench_2", ["regex-log", "other-terminal"]),
    ):
        source = tmp_path / "data/external" / name
        for task_id in task_ids:
            task = source / ("harbor_tasks" if name == "oragentbench" else "") / task_id
            task.mkdir(parents=True)
            (task / "task.toml").write_text('[task]\nname="task"\n')
            if task_id != "family_2":
                (task / "instruction.md").write_text("Solve exact task")
            for file in ("environment/app/data.csv", "tests/test.sh", "solution/solve.sh"):
                path = task / file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
    tau = tmp_path / "data/external/tau3_bench"
    tasks = []
    for index in (1, 2, 3):
        tasks.append(
            {
                "id": str(index),
                "user_scenario": {"instructions": "scenario"},
                "evaluation_criteria": {
                    "actions": [{"arguments": {"user_id": "same-customer"}}],
                    "nl_assertions": ["needs judge"] if index == 3 else [],
                },
            }
        )
    write(tau / "data/tau2/domains/retail/tasks.json", tasks)
    write(
        tau / "data/tau2/domains/retail/split_tasks.json",
        {"train": ["1", "3"], "test": ["2"], "base": ["1", "2", "3"]},
    )
    (tau / ".venv/bin").mkdir(parents=True)
    (tau / ".venv/bin/python").write_text("fixture")
    return tmp_path


@pytest.fixture
def policy():
    return {
        "model": "gpt-6-astra",
        "authentication": "codex_chatgpt_account",
        "reasoning_effort": "xhigh",
        "judge_model": "gpt-5.6-terra",
        "judge_reasoning_effort": "medium",
        "instructions": "SHARED POLICY",
        "timeout_seconds": 60,
        "max_tool_calls": 4,
        "seed": 123,
    }


def test_inventory_keeps_excluded_tasks_and_related_families(dataset_root):
    rows = extended.inventory(dataset_root)
    by_id = {(row["benchmark"], row["id"]): row for row in rows}
    assert len(rows) == 15
    assert all(row["exposed"] for row in rows)
    assert not by_id["putnambench", "putnam_2024_a2"]["adapter_ready"]
    assert not by_id["oragentbench", "family_2"]["adapter_ready"]
    assert not by_id["terminal_bench_2", "other-terminal"]["adapter_ready"]
    assert not by_id["tau3_bench", "retail/3"]["adapter_ready"]
    assert (
        by_id["putnambench", "putnam_2024_a1"]["group_id"]
        == by_id["putnambench", "putnam_2024_a2"]["group_id"]
    )
    assert (
        by_id["harvey_lab", "contracts/test/scenario-01"]["group_id"]
        == by_id["harvey_lab", "contracts/test/scenario-02"]["group_id"]
    )
    assert (
        by_id["tau3_bench", "retail/1"]["group_id"] == by_id["tau3_bench", "retail/2"]["group_id"]
    )
    assert by_id["tau3_bench", "retail/1"]["development_only"]


def test_preflight_resolves_exact_id_and_hashes_actual_inputs(dataset_root):
    one = extended.preflight("gdpval", "gdp-2", root=dataset_root)
    assert one["ready"] and one["structural_preflight_passed"]
    assert one["id"] == "gdp-2" and one["input_file_count"] == 1
    assert one["native_controls_passed"] is None and not one["runtime_validated"]
    (dataset_root / "data/raw/gdpval/reference_files/record-2.txt").write_text("changed input")
    two = extended.preflight("gdpval", "gdp-2", root=dataset_root)
    assert two["input_assets_sha256"] != one["input_assets_sha256"]
    with pytest.raises(ValueError, match="Unknown"):
        extended.preflight("gdpval", "gdp-does-not-exist", root=dataset_root)
    with pytest.raises(ValueError, match="Unknown"):
        extended.preflight("harvey_lab", "../outside", root=dataset_root)


def test_gdp_shared_reference_components_are_transitive():
    rows = [
        {"task_id": "a", "reference_files": ["one"]},
        {"task_id": "b", "reference_files": ["copy-one", "two"]},
        {"task_id": "c", "reference_files": ["two"]},
    ]
    groups = extended._gdp_groups(rows, {"one": "same", "copy-one": "same", "two": "other"})
    assert len(set(groups.values())) == 1


def test_unknown_id_and_unsupported_task_never_dispatch_models(
    monkeypatch, dataset_root, policy, tmp_path
):
    monkeypatch.setattr(extended, "ROOT", dataset_root)
    monkeypatch.setattr(extended, "solve", lambda *a, **k: pytest.fail("Must not call solver"))
    with pytest.raises(ValueError, match="Unknown"):
        extended.evaluate("gdpval", "missing", policy, tmp_path / "out", "image")
    with pytest.raises(extended.UnsupportedTaskError, match="Task-specific"):
        extended.evaluate("terminal_bench_2", "other-terminal", policy, tmp_path / "out", "image")
    assert not (tmp_path / "out").exists()


def test_gdp_evaluation_uses_requested_record_and_private_rubric_only_at_judge(
    monkeypatch, dataset_root, policy, tmp_path
):
    from moevo.codex import document_tasks

    monkeypatch.setattr(extended, "ROOT", dataset_root)
    seen = {}

    def solve(p, prompt, output, image):
        seen["prompt"] = prompt
        seen["reference"] = (output / "workspace/reference_files/record-2.txt").read_text()
        assert not (output / "workspace/reference_files/record-1.txt").exists()
        return CodexResponse("Saved answer")

    def grade(prompt, files, criteria, output, **kwargs):
        seen["rubric"] = criteria
        return {"score": 0.5, "criteria": []}

    monkeypatch.setattr(extended, "solve", solve)
    monkeypatch.setattr(
        document_tasks, "extract", lambda *args: [{"filename": "answer.docx", "text": "work"}]
    )
    monkeypatch.setattr(document_tasks, "gdp_grade", grade)
    result = extended.evaluate("gdpval", "gdp-2", policy, tmp_path / "out", "image")
    assert result["task_id"] == "gdp-2" and result["score"] == 0.5
    assert "Task 2" in seen["prompt"] and "Task 1" not in seen["prompt"]
    assert "private rubric" not in seen["prompt"]
    assert seen["reference"] == "reference 2"
    assert seen["rubric"][0]["criterion"] == "private rubric 2"


def test_health_preserves_native_negative_score_and_jsonl_line_separator(
    monkeypatch, dataset_root, policy, tmp_path
):
    from moevo.codex import health_pilot

    monkeypatch.setattr(extended, "ROOT", dataset_root)
    seen = {}

    def solver(prompt, **kwargs):
        seen["prompt"] = prompt
        assert kwargs["model"] == "gpt-6-astra" and not kwargs["tools"]
        return CodexResponse("answer")

    def grade(row, response, output, **kwargs):
        assert row["id"] == "health-2"
        assert kwargs["judge"]["judge_model"] == "gpt-5.6-terra"
        return {"raw_rubric_score": -0.5, "all_criteria_passed": False}

    monkeypatch.setattr(extended, "run_codex", solver)
    monkeypatch.setattr(health_pilot, "grade_response", grade)
    report = extended.evaluate(
        "healthbench_professional", "health-2", policy, tmp_path / "out", "image"
    )
    assert report["score"] == -0.5 and report["original_score_retained"]
    assert "Medical 2\u2028still one JSONL row" in seen["prompt"]
    assert "SHARED POLICY" in seen["prompt"] and "private rubric" not in seen["prompt"]


def test_legal_unscoped_criteria_receive_all_output_and_actual_task(
    monkeypatch, dataset_root, policy, tmp_path
):
    from moevo.codex import document_tasks

    monkeypatch.setattr(extended, "ROOT", dataset_root)
    seen = []
    monkeypatch.setattr(extended, "solve", lambda *args: CodexResponse("Done"))
    monkeypatch.setattr(
        document_tasks, "extract", lambda *args: [{"filename": "memo.md", "text": "ACTUAL OUTPUT"}]
    )

    def verdict(task, text, title, criterion, output, **kwargs):
        seen.append((task, text))
        return {"verdict": "fail", "reasoning": "Does not meet criterion"}

    monkeypatch.setattr(document_tasks, "legal_verdict", verdict)
    report = extended.evaluate(
        "harvey_lab", "contracts/test/scenario-02", policy, tmp_path / "out", "image"
    )
    assert report["score"] == 0 and report["criterion_pass_rate"] == 0
    assert seen[0][0] == "Legal 2" and "ACTUAL OUTPUT" in seen[0][1]


def test_extraction_errors_are_classified_and_wrong_returned_ids_never_score(
    monkeypatch, dataset_root, policy, tmp_path
):
    monkeypatch.setattr(extended, "ROOT", dataset_root)
    assert extended._check_extraction([{"filename": "bad.pdf", "parse_error": "bad format"}]) == [
        {"filename": "bad.pdf", "parse_error": "bad format"}
    ]
    with pytest.raises(RuntimeError, match="runtime failed"):
        extended._check_extraction(
            [{"filename": "bad.pdf", "parse_error": "No module named pdfplumber"}]
        )
    monkeypatch.setitem(
        extended.HANDLERS,
        "gdpval",
        lambda *args: {"task_id": "gdp-1", "score": 1, "e2e_validated": True},
    )
    with pytest.raises(ValueError, match="different task id"):
        extended.evaluate("gdpval", "gdp-2", policy, tmp_path / "out", "image")
    assert not (tmp_path / "out/report.json").exists()


@pytest.mark.parametrize(
    "benchmark,task_id",
    [
        ("gdpval", "gdp-2"),
        ("harvey_lab", "contracts/test/scenario-02"),
    ],
)
def test_corrupt_candidate_artifacts_remain_scored_task_failures(
    monkeypatch,
    dataset_root,
    policy,
    tmp_path,
    benchmark,
    task_id,
):
    from moevo.codex import document_tasks

    monkeypatch.setattr(extended, "ROOT", dataset_root)
    monkeypatch.setattr(extended, "solve", lambda *args: CodexResponse("Saved corrupt file"))
    monkeypatch.setattr(
        document_tasks,
        "extract",
        lambda *args: [
            {"filename": "answer.docx", "parse_error": "File is not a zip file"},
        ],
    )
    monkeypatch.setattr(
        document_tasks,
        "gdp_grade",
        lambda *a, **k: pytest.fail("Invalid files must not be graded as evidence"),
    )
    monkeypatch.setattr(
        document_tasks,
        "legal_verdict",
        lambda *a, **k: pytest.fail("Invalid files must not be graded as evidence"),
    )
    report = extended.evaluate(benchmark, task_id, policy, tmp_path / "out", "image")
    assert report["score"] == 0.0 and report["e2e_validated"]
    assert report["grade"]["invalid_submission"] is True
    assert report["grade"]["invalid_artifacts"][0]["filename"] == "answer.docx"
    assert (tmp_path / "out/report.json").exists()


def test_or_missing_solution_retains_native_zero(tmp_path):
    (tmp_path / "reward.txt").write_text("0\n")
    scored = {
        "evaluation": {"feasible": False, "errors": ["No solution file found."]},
        "reward": {"feasibility": 0.0, "quality": 0.0, "quality_status": "missing_solution"},
    }
    assert extended._or_scalar_reward(scored, tmp_path) == 0.0
    assert "scalar_reward" not in scored["reward"]  # Native evidence stays unchanged.


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s["reward"].update(quality_status="infeasible"),
        lambda s: s["reward"].update(quality=1.0),
        lambda s: s["reward"].update(feasibility=False),
        lambda s: s["evaluation"].update(feasible=True),
        lambda s: s["evaluation"].update(errors=["Could not parse evaluator JSON: failed"]),
    ],
)
def test_or_missing_scalar_is_not_an_arbitrary_zero(tmp_path, mutation):
    (tmp_path / "reward.txt").write_text("0\n")
    scored = {
        "evaluation": {"feasible": False},
        "reward": {"feasibility": 0.0, "quality": 0.0, "quality_status": "missing_solution"},
    }
    mutation(scored)
    with pytest.raises((ValueError, RuntimeError)):
        extended._or_scalar_reward(scored, tmp_path)


def test_or_explicit_score_must_match_native_file_and_formula(tmp_path):
    (tmp_path / "reward.txt").write_text("0.5\n")
    scored = {
        "evaluation": {"feasible": True},
        "reward": {
            "feasibility": 1.0,
            "quality": 0.5,
            "scalar_reward": 0.5,
            "quality_status": "scored_with_mip_gap_fallback",
        },
    }
    assert extended._or_scalar_reward(scored, tmp_path) == 0.5
    (tmp_path / "reward.txt").write_text("1\n")
    with pytest.raises(ValueError, match="disagree"):
        extended._or_scalar_reward(scored, tmp_path)


def test_existing_artifacts_and_api_policy_are_rejected(
    monkeypatch, dataset_root, policy, tmp_path
):
    monkeypatch.setattr(extended, "ROOT", dataset_root)
    output = tmp_path / "out"
    output.mkdir()
    (output / "evidence.txt").write_text("preserve me")
    with pytest.raises(FileExistsError):
        extended.evaluate("gdpval", "gdp-1", policy, output, "image")
    assert (output / "evidence.txt").read_text() == "preserve me"
    with pytest.raises(ValueError, match="account Astra"):
        extended.evaluate(
            "gdpval", "gdp-1", {**policy, "authentication": "api_key"}, tmp_path / "other", "image"
        )
