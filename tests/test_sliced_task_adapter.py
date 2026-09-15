"""Frozen task and partition boundaries must hold before any model invocation."""

import json
from types import SimpleNamespace

import pytest

from experiments import sliced_task_adapter as adapter


def setup_run(tmp_path, monkeypatch, role="search"):
    task = {
        "id": "finqa",
        "benchmark": "finqa",
        "task_id": "original-id",
        "objective": "finance",
        "seed": 42,
        "content_sha256": "expected",
    }
    panel = tmp_path / "search.json"
    panel.write_text(json.dumps({"split": role, "tasks": [task]}))
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "panel_path": str(panel),
                "execution_role": role,
                "runtime_image": "pinned",
                "policy": {
                    "authentication": "codex_chatgpt_account",
                    "model": "gpt-6-astra",
                    "judge_model": "gpt-5.6-terra",
                },
            }
        )
    )
    program = tmp_path / "seed.py"
    program.write_text("INSTRUCTIONS = 'Solve and verify.'")
    monkeypatch.setenv("MOEVO_SLICED_DIR", str(tmp_path))
    return program, task


def test_final_and_changed_task_identity_never_reach_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "backend", lambda _: pytest.fail("No backend should load"))
    program, task = setup_run(tmp_path, monkeypatch, "final")
    with pytest.raises(ValueError, match="Final evaluation"):
        adapter.evaluate_task(str(program), task)
    program, task = setup_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="identity"):
        adapter.evaluate_task(str(program), {**task, "task_id": "different-task"})


def test_changed_assets_block_inference_and_native_zero_is_retained(tmp_path, monkeypatch):
    program, task = setup_run(tmp_path, monkeypatch)
    calls = []
    check = {"structural_preflight_passed": True, "content_sha256": "changed"}

    def evaluate(*args):
        calls.append(args)
        return {
            "task_id": "original-id",
            "e2e_validated": True,
            "score": 0.0,
            "grade": {"reason": "incorrect answer", "correct_answer": "PRIVATE KEY"},
        }

    monkeypatch.setattr(
        adapter, "backend", lambda _: SimpleNamespace(preflight=lambda *a: check, evaluate=evaluate)
    )
    with pytest.raises(ValueError, match="content changed"):
        adapter.evaluate_task(str(program), task)
    assert calls == []
    check["content_sha256"] = "expected"
    result = adapter.evaluate_task(str(program), task)
    assert result["status"] == "scored" and result["score"] == 0
    assert result["native_metrics"]["grade"]["correct_answer"] == "PRIVATE KEY"
    assert "PRIVATE KEY" not in result["feedback"]
