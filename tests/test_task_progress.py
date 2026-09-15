"""Progress must follow actual dispatch and saved scores, not optimistic queues."""

import hashlib
import json

from moevo.reporting.progress import controller_liveness, screen_task_ids, task_progress
from moevo.reporting.source import code_hash

SEED = 'INSTRUCTIONS = "seed"\n'
CHILD = 'INSTRUCTIONS = "child"\n'


def fixture(tmp_path, count=4):
    (tmp_path / "seed.py").write_text(SEED)
    tasks = [
        {"id": str(i), "benchmark": f"b{i}", "objective": f"d{i}", "task_id": f"published-{i}"}
        for i in range(count)
    ]
    manifest = {"tasks": tasks, "confirmation_ids": [t["id"] for t in tasks]}
    run = {
        "status": "running",
        "config": {
            "objectives": [t["objective"] for t in tasks],
            "random_seed": 42,
            "screen_domains": 2,
            "screen_tasks_per_domain": 1,
        },
    }
    state = {"cache": {}, "screens": 0, "events": [], "task_evaluations": 0}
    return run, manifest, state


def attempt(tmp_path, task, source=SEED):
    prefix = hashlib.sha256(source.encode()).hexdigest()[:12]
    path = tmp_path / "tasks" / f"b{task}" / f"{prefix}-12345678"
    path.mkdir(parents=True)
    (path / "preflight.json").write_text(json.dumps({"id": f"published-{task}"}))
    return path


def save_score(tmp_path, state, task, score=1, source=SEED):
    path = attempt(tmp_path, task, source)
    state["cache"][source + task] = {
        "task_id": task,
        "candidate_sha256": code_hash(source),
        "score": score,
        "usage": {"artifact_path": str(path.relative_to(tmp_path))},
    }


def test_zero_is_scored_and_free_slot_does_not_start_next_wave(tmp_path):
    run, manifest, state = fixture(tmp_path)
    save_score(tmp_path, state, "1", score=0)
    attempt(tmp_path, "0")
    state["task_evaluations"] = 2
    p = task_progress(tmp_path, run, manifest, state, {}, process={"state": "active"})
    assert [r["status"] for r in p["tasks"]] == ["running", "scored", "queued", "queued"]
    assert p["stage_scored"] == p["unique_tasks_scored"] == 1
    assert p["stage_running"] == 1 and p["attempts_charged"] == 2
    assert p["tasks"][1]["score"] == 0


def test_parent_cache_does_not_fill_child_progress(tmp_path):
    run, manifest, state = fixture(tmp_path)
    for task in manifest["tasks"]:
        save_score(tmp_path, state, task["id"])
    state["screens"] = 1
    panel = screen_task_ids(manifest, run["config"], 0)
    save_score(tmp_path, state, panel[0], source=CHILD)
    path = tmp_path / "source_observations"
    path.mkdir()
    (path / f"{code_hash(CHILD)}.py").write_text(CHILD)
    attempt(tmp_path, panel[1], CHILD)
    p = task_progress(tmp_path, run, manifest, state, {}, process={"state": "active"})
    assert p["phase"] == "quick_check" and p["stage_scored"] == 1
    assert p["stage_total"] == 2 and p["unique_tasks_scored"] == 4
    assert sum(r["status"] == "not_in_check" for r in p["tasks"]) == 2
    assert p["stage_running"] == 1


def test_first_child_attempt_visible_before_first_cached_score(tmp_path):
    run, manifest, state = fixture(tmp_path)
    for task in manifest["tasks"]:
        save_score(tmp_path, state, task["id"])
    state["screens"] = 1
    panel = screen_task_ids(manifest, run["config"], 0)
    attempt(tmp_path, panel[0], CHILD)
    p = task_progress(tmp_path, run, manifest, state, {}, process={"state": "active"})
    assert p["candidate_sha256"] is None
    assert p["stage_scored"] == 0 and p["stage_running"] == 1 and p["stage_queued"] == 1


def test_report_pending_recording_does_not_become_score(tmp_path):
    run, manifest, state = fixture(tmp_path)
    path = attempt(tmp_path, "0")
    (path / "report.json").write_text('{"score": 1}')
    p = task_progress(tmp_path, run, manifest, state, {}, process={"state": "active"})
    assert p["stage_scored"] == 0
    assert p["tasks"][0]["detail"] == "Saving result"


def test_stale_running_marker_and_error_are_not_live_tasks(tmp_path):
    run, manifest, state = fixture(tmp_path)
    attempt(tmp_path, "0")
    path = attempt(tmp_path, "1")
    (path / "error.json").write_text('{"error": "transport failed"}')
    p = task_progress(tmp_path, run, manifest, state, {}, process={"state": "stopped"})
    assert p["stage_running"] == p["stage_scored"] == 0
    assert p["tasks"][0]["status"] == "unfinished"
    assert p["tasks"][1]["status"] == "error" and p["tasks"][1]["score"] is None
    p = task_progress(tmp_path, run, manifest, state, {})
    assert p["tasks"][0]["status"] == "started"


def test_full_test_counts_cached_quick_scores_once(tmp_path):
    run, manifest, state = fixture(tmp_path)
    save_score(tmp_path, state, "0", source=CHILD)
    state["screens"] = 1
    state["events"] = [
        {
            "stage": "screen",
            "screen": 1,
            "task_ids": ["0", "1"],
            "candidate_sha256": code_hash(CHILD),
            "improved": True,
            "confirmed": False,
        }
    ]
    p = task_progress(tmp_path, run, manifest, state, {})
    assert p["phase"] == "full_test" and p["stage_total"] == 4 and p["stage_scored"] == 1


def test_ambiguous_same_benchmark_attempt_is_not_assigned_to_wrong_task(tmp_path):
    run, manifest, state = fixture(tmp_path, 2)
    manifest["tasks"][1]["benchmark"] = "b0"
    path = attempt(tmp_path, "0")
    (path / "preflight.json").unlink()
    p = task_progress(tmp_path, run, manifest, state, {}, process={"state": "active"})
    assert p["stage_running"] == 0
    (path / "preflight.json").write_text('{"id":"published-1"}')
    p = task_progress(tmp_path, run, manifest, state, {}, process={"state": "active"})
    assert p["tasks"][1]["status"] == "running" and p["tasks"][0]["status"] == "queued"


def test_process_check_matches_controller_and_exact_output(tmp_path, monkeypatch):
    from types import SimpleNamespace

    text = (
        f'1 python -m experiments.evolution_dashboard --run "{tmp_path}"\n'
        f'2 python -m experiments.run_first_slice --output "{tmp_path}-other"\n'
        f'3 python -m experiments.run_first_slice --output "{tmp_path}"\n'
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **k: SimpleNamespace(stdout=text))
    assert controller_liveness(tmp_path) == {"state": "active", "pids": [3]}


def test_new_attempt_wins_over_an_older_failed_partial_candidate(tmp_path):
    run, manifest, state = fixture(tmp_path)
    for task in manifest["tasks"]:
        save_score(tmp_path, state, task["id"])
    state["screens"] = 2
    panel = screen_task_ids(manifest, run["config"], 1)
    save_score(tmp_path, state, panel[0], source=CHILD)
    observed = tmp_path / "source_observations"
    observed.mkdir()
    (observed / f"{code_hash(CHILD)}.py").write_text(CHILD)
    # No screen event was committed for a previously failed partial candidate.
    newest = 'INSTRUCTIONS = "newest"\n'
    attempt(tmp_path, panel[0], newest)
    p = task_progress(tmp_path, run, manifest, state, {}, process={"state": "active"})
    assert p["candidate_sha256"] is None and p["stage_scored"] == 0
    assert p["stage_running"] == 1
