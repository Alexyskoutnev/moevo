"""Model-role separation and preservation of historical judge protocols."""

import json
from pathlib import Path

import pytest

from experiments.run_submission import load_submission
from moevo.codex.client import CodexResponse
from moevo.codex.document_tasks import gdp_grade
from moevo.codex.judging import judge_options, run_judge

ROOT = Path(__file__).resolve().parents[1]


def test_new_pilot_keeps_astra_solver_and_separate_terra_judge(monkeypatch, tmp_path):
    policy = load_submission(ROOT / "submissions/superharness-astra-terra-pilot/submission.json")
    assert policy["model"] == "gpt-6-astra" and policy["reasoning_effort"] == "xhigh"
    assert judge_options(policy) == {"model": "gpt-5.6-terra", "effort": "medium"}
    seen = []

    def fake(prompt, **kwargs):
        seen.append(kwargs)
        return CodexResponse(text="true")

    monkeypatch.setattr("moevo.codex.judging.run_codex", fake)
    assert run_judge("Grade only", judge=policy, cwd=tmp_path).text == "true"
    assert seen[0]["model"] == "gpt-5.6-terra" and seen[0]["effort"] == "medium"
    assert policy["model"] == "gpt-6-astra"


def test_existing_manifest_remains_astra_judged():
    old = load_submission(ROOT / "submissions/superharness-astra-v1/submission.json")
    assert judge_options(old) == {"model": "gpt-6-astra", "effort": "xhigh"}


@pytest.mark.parametrize(
    "policy",
    [
        {"judge_model": "gpt-5.6-terra"},
        {"judge_model": "openai/gpt-5-mini", "judge_reasoning_effort": "medium"},
        {"judge_model": "gpt-5.6-terra", "judge_reasoning_effort": "invalid"},
    ],
)
def test_incomplete_or_nonaccount_routing_fails(policy):
    with pytest.raises(ValueError):
        judge_options(policy)


@pytest.mark.parametrize(
    "verdict",
    [
        {"criteria": [{"id": 0, "met": "false", "reasoning": "No"}]},
        {"criteria": [{"id": False, "met": False, "reasoning": "No"}]},
        {"criteria": []},
    ],
)
def test_gdp_rejects_malformed_judgment_instead_of_awarding_points(monkeypatch, tmp_path, verdict):
    monkeypatch.setattr(
        "moevo.codex.judging.run_codex",
        lambda *args, **kwargs: CodexResponse(text=json.dumps(verdict)),
    )
    with pytest.raises(ValueError, match="omitted or duplicated"):
        gdp_grade("Task", [], [{"score": 2, "criterion": "Required"}], tmp_path)
