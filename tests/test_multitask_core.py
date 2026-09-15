"""Exact IDs, input isolation, and honest preflight status for multi-task adapters."""

import json
from types import SimpleNamespace

import pytest

from moevo.codex import multitask_core as core


def _finqa_row(task_id):
    return {
        "id": task_id,
        "filename": task_id.split(".pdf")[0] + ".pdf",
        "pre_text": ["Revenue is 10"],
        "post_text": [],
        "table": [["Revenue", "10"]],
        "qa": {"question": "What is revenue?", "program": "gold secret", "exe_ans": 10},
    }


def _finqa_root(tmp_path):
    root = tmp_path / "repo"
    folder = root / "data/raw/finqa"
    (folder / "dataset").mkdir(parents=True)
    rows = [_finqa_row("ABC/2020/page_1.pdf-1"), _finqa_row("ABC/2020/page_1.pdf-2")]
    (folder / "dataset/train.json").write_text(json.dumps(rows))
    (folder / "manifest.json").write_text(
        json.dumps({"revision": "pinned", "repository": "upstream"})
    )
    return root, rows


def test_inventory_groups_same_report_and_hashes_public_inputs_without_gold(tmp_path):
    root, _ = _finqa_root(tmp_path)
    rows = core.inventory(["finqa"], root=root)
    assert len(rows) == 2
    assert rows[0]["group_id"] == rows[1]["group_id"] == "finqa:ABC/2020"
    assert rows[0]["content_sha256"] == rows[1]["content_sha256"]
    assert all(row["exposed"] for row in rows)
    assert "gold secret" not in json.dumps(rows)
    assert rows[0]["source"]["dataset_revision"] == "pinned"


def test_preflight_rejects_unknown_id_instead_of_using_seed_fixture(tmp_path):
    root, _ = _finqa_root(tmp_path)
    with pytest.raises(KeyError, match="Unknown finqa task ID"):
        core.preflight("finqa", "not-present", root=root)


def test_structural_preflight_does_not_claim_execution_or_grader_success(tmp_path):
    root, rows = _finqa_root(tmp_path)
    result = core.preflight("finqa", rows[1]["id"], root=root)
    assert result["task_id"] == rows[1]["id"]
    assert result["structural_preflight_passed"]
    assert result["native_controls_passed"] is None
    assert result["model_calls"] == 0
    assert result["e2e_validated"] is False


def test_failing_annotation_is_preserved_and_does_not_trigger_task_resampling(
    tmp_path, monkeypatch
):
    root, rows = _finqa_root(tmp_path)
    called = []

    def grade(benchmark, row, prediction, output, image, source_root):
        called.append(row["id"])
        return {"score": 0}

    monkeypatch.setattr(core, "_native_grade", grade)
    monkeypatch.setattr(core, "_control_answers", lambda *args: ("gold", "empty"))
    output = tmp_path / "controls"
    with pytest.raises(ValueError, match="task not replaced"):
        core.preflight("finqa", rows[1]["id"], output, native_controls=True, root=root)
    assert called == [rows[1]["id"], rows[1]["id"]]
    assert json.loads((output / "preflight.json").read_text())["native_controls_passed"] is False


def test_genebench_rejects_answer_or_traversal_paths(tmp_path):
    row = {"data_files": ["../eval_config.json"], "task": "scientific task"}
    with pytest.raises(ValueError, match="Invalid GeneBench"):
        core._public("genebench_pro", "case", row, tmp_path)


def test_dsbench_copies_only_public_data_and_selected_question(tmp_path):
    folder = tmp_path / "data/raw/dsbench/processed/data/group1"
    folder.mkdir(parents=True)
    for name in ["input.csv", "introduction.txt", "question1.txt", "question2.txt", "answers.txt"]:
        (folder / name).write_text(name)
    prompt, inputs = core._public(
        "dsbench",
        "group1/question1",
        {
            "id": "group1",
            "question_name": "question1",
            "answers": ["secret answer"],
        },
        tmp_path,
    )
    assert {name for _, name in inputs} == {"input.csv", "introduction.txt", "question.txt"}
    assert "question1.txt" in prompt
    assert "secret answer" not in prompt
    assert "question2" not in prompt


def test_dsbench_rejects_symlink_to_files_outside_group(tmp_path):
    folder = tmp_path / "data/raw/dsbench/processed/data/group1"
    folder.mkdir(parents=True)
    outside = tmp_path / "secret.csv"
    outside.write_text("do not send")
    (folder / "input.csv").symlink_to(outside)
    with pytest.raises(ValueError, match="Invalid or missing public input"):
        core._public(
            "dsbench", "group1/question1", {"id": "group1", "question_name": "question1"}, tmp_path
        )


def _policy():
    return {
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "authentication": "codex_chatgpt_account",
        "instructions": "Shared harness policy",
        "judge_model": "gpt-5.6-terra",
        "judge_reasoning_effort": "medium",
        "timeout_seconds": 600,
        "max_tool_calls": 20,
    }


def test_evaluate_uses_exact_requested_task_and_separates_control_workspace(tmp_path, monkeypatch):
    from moevo.codex import domain_tasks

    called = []
    monkeypatch.setattr(
        core,
        "preflight",
        lambda benchmark, task_id, *args, **kwargs: {
            "task_id": task_id,
            "content_sha256": "abc",
            "source": {},
            "controls": {},
        },
    )
    monkeypatch.setattr(core, "_lookup", lambda benchmark, task_id, root: {"id": task_id})
    monkeypatch.setattr(
        core, "_public", lambda benchmark, task_id, row, root: ("Task " + task_id, [])
    )

    def solve(policy, prompt, output, image):
        assert policy["model"] == "gpt-6-astra"
        assert list((output / "workspace").iterdir()) == []
        called.append(prompt)
        return SimpleNamespace(text="wrong answer", usage={"input_tokens": 1}, duration_s=2)

    monkeypatch.setattr(domain_tasks, "solve", solve)
    monkeypatch.setattr(
        core,
        "_native_grade",
        lambda benchmark, row, *args: {"score": 0, "observed_task_id": row["id"]},
    )
    result = core.evaluate("finqa", "requested-task-two", _policy(), tmp_path / "task")
    assert called == ["Task requested-task-two"]
    assert result["task_id"] == result["grade"]["observed_task_id"] == "requested-task-two"
    assert result["score"] == 0
    assert result["e2e_validated"]


def test_existing_solver_workspace_and_wrong_model_are_rejected(tmp_path):
    folder = tmp_path / "workspace"
    folder.mkdir()
    (folder / "old-answer.txt").write_text("contamination")
    with pytest.raises(ValueError, match="nonempty"):
        core.evaluate("finqa", "anything", _policy(), tmp_path)
    with pytest.raises(ValueError, match="gpt-6-astra"):
        core.evaluate("finqa", "anything", {**_policy(), "model": "old-model"}, tmp_path)


def test_bizfin_prompt_excludes_assistant_gold():
    prompt = core._biz_prompt(
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Compute revenue."}]},
                {"role": "assistant", "content": "SECRET_GOLD"},
            ]
        }
    )
    assert "Compute revenue." in prompt
    assert "SECRET_GOLD" not in prompt
