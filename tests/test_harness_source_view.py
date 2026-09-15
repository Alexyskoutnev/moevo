"""Source display must show actual evaluated bytes and distinguish unconfirmed code."""

import json

from experiments.capture_evolution_sources import capture
from moevo.reporting.source import code_hash, source_view


def test_capture_only_sources_linked_to_this_runs_ledger(tmp_path):
    run, temporary = tmp_path / "run", tmp_path / "temporary"
    run.mkdir()
    seed = "INSTRUCTIONS = 'Solve the task.'\n"
    child = "INSTRUCTIONS = 'Solve the task. Verify the output.'\n"
    (run / "seed.py").write_text(seed)
    candidate = temporary / "moevo-tasks-current" / "candidate.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(child)
    unrelated = temporary / "moevo-tasks-unrelated" / "candidate.py"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("INSTRUCTIONS = 'Unrelated private experiment.'")
    state = {"cache": {"key": {"candidate_sha256": code_hash(child), "task_id": "one"}}}
    (run / "evaluation_state.json").write_text(json.dumps(state))
    assert capture(run, temporary) == 1
    candidate.unlink()
    data = source_view(run, {}, state)
    assert len(data["versions"]) == 2
    changed = data["versions"][1]
    assert changed["status"] == "evaluation_in_progress"
    assert changed["code"] == child
    assert "-INSTRUCTIONS = 'Solve the task.'" in changed["diff"]
    assert "+INSTRUCTIONS = 'Solve the task. Verify the output.'" in changed["diff"]
    assert "Unrelated private experiment" not in json.dumps(data)
    assert "Verify the output" in changed["instructions"]


def test_corrupted_observation_and_rejected_candidate_are_not_promoted(tmp_path):
    seed = "INSTRUCTIONS = 'Solve.'\n"
    child = "INSTRUCTIONS = 'Check.'\n"
    (tmp_path / "seed.py").write_text(seed)
    directory = tmp_path / "source_observations"
    directory.mkdir()
    sha = code_hash(child)
    path = directory / f"{sha}.py"
    path.write_text(child)
    state = {
        "cache": {"one": {"candidate_sha256": sha, "task_id": "x"}},
        "events": [
            {
                "stage": "screen",
                "candidate_sha256": sha,
                "screen": 1,
                "improved": False,
                "audit": False,
            }
        ],
    }
    assert source_view(tmp_path, {}, state)["versions"][1]["status"] == "screened_out"
    path.write_text("INSTRUCTIONS = 'Tampered.'")
    assert len(source_view(tmp_path, {}, state)["versions"]) == 1
