"""Safe coordinator fencing, adoption, parallel bounds and judge-binding repair."""

import fcntl
import signal
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from experiments import run_reference_baselines as frozen
from experiments import run_reference_parallel as parallel


@pytest.fixture
def ledger(tmp_path):
    tasks = [
        {
            "id": name,
            "benchmark": "finqa",
            "task_id": "native-" + name,
            "objective": "finance",
            "seed": 1,
        }
        for name in ("a", "b", "c", "d")
    ]
    run = {
        "identity_sha256": "identity",
        "baseline_id": "codex_cli_task_only",
        "workers": 1,
        "tasks": tasks,
        "max_task_attempts": 12,
        "worker_timeout_seconds": 5400,
        "cli_version": "test-cli",
    }
    frozen.write_json(tmp_path / "run.json", run)
    frozen.write_json(
        tmp_path / "reference_state.json",
        {
            "task_attempts": 0,
            "model_calls": 0,
            "calls": {},
            "tasks": {
                task["id"]: {
                    **task,
                    "status": "pending",
                    "score": None,
                    "native_score": None,
                    "attempts": [],
                    "error": None,
                }
                for task in tasks
            },
        },
    )
    amendment = tmp_path / "execution_amendments/a.json"
    frozen.write_json(
        amendment,
        {"id": "a", "coordinator": {"pid": 1234}, "started_utc": frozen.utcnow(), "events": []},
    )
    return tmp_path, run, amendment


def coordinator(output, state="S"):
    return {
        "pid": 1234567,
        "start_identity": "Tue Sep 15 15:00:00 2026",
        "state": state,
        "argv": [
            "python",
            "-m",
            "experiments.run_reference_baselines",
            "--output",
            str(output),
            "--resume",
        ],
    }


def test_fence_stops_only_verified_coordinator_pid(ledger, monkeypatch):
    output, _, amendment = ledger
    current, sent = coordinator(output), []
    monkeypatch.setattr(parallel, "process_identity", lambda pid: dict(current))

    def kill(pid, sig):
        sent.append((pid, sig))
        if sig == signal.SIGSTOP:
            current["state"] = "T"

    monkeypatch.setattr(parallel.os, "kill", kill)
    parallel.fence_serial(current["pid"], output, amendment)
    assert sent == [(current["pid"], signal.SIGSTOP), (current["pid"], signal.SIGKILL)]
    assert (
        frozen.read_json(amendment)["events"][-1]["event"]
        == "serial_coordinator_stopped_workers_preserved"
    )


def test_failed_fence_resumes_verified_parent_without_killing(ledger, monkeypatch):
    output, _, amendment = ledger
    expected = coordinator(output)
    observations = iter([expected, {**expected, "start_identity": "changed"}, expected])
    sent = []
    monkeypatch.setattr(parallel, "process_identity", lambda pid: next(observations))
    monkeypatch.setattr(parallel.os, "kill", lambda pid, sig: sent.append(sig))
    with pytest.raises(RuntimeError, match="identity changed"):
        parallel.fence_serial(expected["pid"], output, amendment)
    assert sent == [signal.SIGSTOP, signal.SIGCONT]


def test_wrong_output_or_worker_pid_is_not_a_coordinator(ledger):
    output, _, _ = ledger
    with pytest.raises(ValueError, match="different output"):
        parallel.verify_coordinator(coordinator(output / "other"), output)
    identity = coordinator(output)
    identity["argv"] += ["--_worker", "finqa"]
    with pytest.raises(ValueError, match="not the frozen serial coordinator"):
        parallel.verify_coordinator(identity, output)


def test_worker_lease_before_pid_registration_counts_as_active(ledger):
    output, run, _ = ledger
    attempt = frozen.charge_task(output, run, "a")
    assert attempt is not None
    destination = output / attempt["output"]
    destination.mkdir(parents=True)
    with (destination / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert parallel.reconcile_task(output, run, "a", {}) == "active"
    assert frozen.read_json(output / "reference_state.json")["task_attempts"] == 1


def commit_result(output, run, task_id, attempt):
    task = next(task for task in run["tasks"] if task["id"] == task_id)
    frozen.write_json(
        output / attempt["output"] / "output.json",
        {
            "identity_sha256": run["identity_sha256"],
            "attempt_id": attempt["id"],
            "task": task,
            "status": "scored",
            "score": 0.0,
            "native_score": 0.0,
        },
    )


def test_committed_output_is_adopted_once_without_recharging(ledger, monkeypatch):
    output, run, _ = ledger
    attempt = frozen.charge_task(output, run, "a")
    commit_result(output, run, "a", attempt)
    monkeypatch.setattr(frozen, "audit_attempt", lambda *args: {})
    assert parallel.reconcile_task(output, run, "a", {}) == "settled"
    assert parallel.reconcile_task(output, run, "a", {}) == "settled"
    state = frozen.read_json(output / "reference_state.json")
    assert state["task_attempts"] == 1 and state["tasks"]["a"]["score"] == 0.0


def test_unsettled_model_call_never_launches_replacement(ledger, monkeypatch):
    output, run, _ = ledger
    attempt = frozen.charge_task(output, run, "a")
    assert attempt is not None
    with frozen.state_transaction(output) as state:
        state["calls"]["call"] = {"attempt_id": attempt["id"], "status": "running"}
        state["tasks"]["a"]["attempts"][-1]["worker"] = {"pid": 999999, "start_identity": "old"}
    monkeypatch.setattr(parallel, "live_registered_worker", lambda attempt: False)
    with pytest.raises(RuntimeError, match="Unsettled model call"):
        parallel.reconcile_task(output, run, "a", {})
    state = frozen.read_json(output / "reference_state.json")
    assert state["task_attempts"] == 1 and state["tasks"]["a"]["status"] == "running"


def test_adoption_keeps_original_deadline_and_timeout_is_unavailable(ledger, monkeypatch):
    output, run, _ = ledger
    frozen.charge_task(output, run, "a")
    with frozen.state_transaction(output) as state:
        attempt = state["tasks"]["a"]["attempts"][-1]
        attempt["started_utc"] = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        attempt["worker"] = {"pid": 999999, "start_identity": "old"}
    stopped = []
    monkeypatch.setattr(parallel, "stop_expired_worker", lambda *args: stopped.append(args[1]))
    monkeypatch.setattr(parallel, "live_registered_worker", lambda attempt: False)
    assert parallel.reconcile_task(output, run, "a", {}) == "settled"
    assert stopped == ["a"]
    row = frozen.read_json(output / "reference_state.json")["tasks"]["a"]
    assert row["status"] == "error" and row["score"] is None
    assert "original wall-clock deadline" in row["error"]


@pytest.mark.parametrize("completed_seconds,expected_status", [(100, "scored"), (5500, "error")])
def test_late_reconciliation_preserves_only_results_completed_before_deadline(
    ledger, monkeypatch, completed_seconds, expected_status
):
    output, run, _ = ledger
    frozen.charge_task(output, run, "a")
    started = datetime.now(UTC) - timedelta(hours=2)
    with frozen.state_transaction(output) as state:
        attempt = state["tasks"]["a"]["attempts"][-1]
        attempt["started_utc"] = started.isoformat()
        attempt["worker"] = {"pid": 999999, "start_identity": "old"}
    commit_result(output, run, "a", attempt)
    path = output / attempt["output"] / "output.json"
    result = frozen.read_json(path)
    result["completed_utc"] = (started + timedelta(seconds=completed_seconds)).isoformat()
    frozen.write_json(path, result)
    stops = []
    monkeypatch.setattr(parallel, "stop_expired_worker", lambda *args: stops.append(True))
    monkeypatch.setattr(parallel, "live_registered_worker", lambda attempt: False)
    monkeypatch.setattr(frozen, "audit_attempt", lambda *args: {})
    assert parallel.reconcile_task(output, run, "a", {}) == "settled"
    row = frozen.read_json(output / "reference_state.json")["tasks"]["a"]
    assert row["status"] == expected_status
    assert stops == ([] if expected_status == "scored" else [True])


def test_rolling_pool_includes_adopted_attempt_and_never_repeats_scores(ledger, monkeypatch):
    from moevo.codex import client

    output, run, amendment = ledger
    frozen.charge_task(output, run, "a")
    monkeypatch.setattr(client, "require_chatgpt_login", lambda: "test-cli")
    seen, peak, polls = [], [], {}

    def launch(out, cfg, task_id, amendment, retry_error=False):
        assert task_id not in seen and task_id != "a"
        seen.append(task_id)
        frozen.charge_task(out, cfg, task_id)
        state = frozen.read_json(out / "reference_state.json")
        peak.append(sum(row["status"] == "running" for row in state["tasks"].values()))
        return SimpleNamespace(pid=12345, wait=lambda timeout: 0)

    def reconcile(out, cfg, task_id, owned):
        polls[task_id] = polls.get(task_id, 0) + 1
        if polls[task_id] < 2:
            return "active"
        attempt = frozen.read_json(out / "reference_state.json")["tasks"][task_id]["attempts"][-1]
        commit_result(out, cfg, task_id, attempt)
        frozen.finish_attempt(out, cfg, task_id, attempt)
        return "settled"

    monkeypatch.setattr(parallel, "launch_task", launch)
    monkeypatch.setattr(parallel, "reconcile_task", reconcile)
    state = parallel.coordinate(output, run, amendment, 3, poll_seconds=0)
    assert seen == ["b", "c", "d"]
    assert max(peak) == 3 and state["task_attempts"] == 4
    assert state["scored_tasks"] == 4 and state["status"] == "completed"
    assert frozen.read_json(output / "run.json")["workers"] == 1


def test_only_pre_repair_guard_error_is_retryable():
    row = {
        "status": "error",
        "error": "Uncharged Codex command blocked",
        "attempts": [{"id": "old"}],
    }
    assert parallel.known_audit_guard_failure(row)
    row["attempts"][-1]["worker_transport"] = parallel.WORKER_PROTOCOL
    assert not parallel.known_audit_guard_failure(row)
    row["attempts"][-1].pop("worker_transport")
    row["error"] = "Native grader infrastructure failed"
    assert not parallel.known_audit_guard_failure(row)


def test_fresh_import_judge_alias_repair_routes_audited_account_call(tmp_path):
    # Reproduce the real package initialization order in a clean interpreter.
    script = r"""
import json, sys
from pathlib import Path
from experiments import run_reference_baselines as ref, reference_worker_v2 as shim
from moevo.codex import client, judging
out=Path(sys.argv[1])
assert judging.run_codex is client.run_codex
def fake(prompt, **kwargs):
    client.build_command(out,out/'answer',kwargs['model'],kwargs['effort'],False)
    return client.CodexResponse('true',usage={'output_tokens':1})
client.run_codex=fake
assert judging.run_codex is not client.run_codex
run={'adapter_policy':{'instructions':'Seed prefix'},'baseline_id':'codex_cli_task_only','runtime_images':{},'max_model_calls':2}
ref.write_json(out/'reference_state.json',{'model_calls':0,'calls':{},'model_usage':{},'tasks':{'dsbench':{'status':'running','attempts':[{'id':'attempt-1','output':'tasks/ds/attempt-1'}]}}})
shim.install_binding_repair()
ref.install_worker_hooks(out,run,'dsbench','attempt-1')
assert judging.run_codex is client.run_codex
judging.run_judge('Rubric remains unchanged',judge={'judge_model':'gpt-5.6-terra','judge_reasoning_effort':'medium'},cwd=out,tools=False)
state=ref.read_json(out/'reference_state.json')
assert state['model_calls']==1
assert state['calls']['call-00000']['role']=='judge'
assert state['calls']['call-00000']['unchanged'] is True
assert (out/'model_calls/call-00000/command.json').is_file()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
