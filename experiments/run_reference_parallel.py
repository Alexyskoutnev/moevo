"""Explicit concurrency amendment for already-frozen reference evaluations.

Keep the original evaluator, task/model budgets and identity unchanged. Adopt
in-flight workers, and launch pending tasks through the frozen worker entrypoint.
An optional takeover fences ONLY the verified serial coordinator; task workers
continue. Existing scored/error attempts are never rerun.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments import run_reference_baselines as frozen
from experiments.reference_worker_v2 import PROTOCOL as WORKER_PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "reference-parallel-coordinator-amendment-v1"


def process_identity(pid: int) -> dict | None:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=", "-o", "stat=", "-o", "args="],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode == 1 and not result.stdout.strip():
        return None
    if result.returncode:
        raise RuntimeError("Cannot verify process identity: " + result.stderr[-500:])
    fields = result.stdout.strip().split(maxsplit=6)
    if len(fields) != 7:
        raise ValueError("Unexpected process identity format")
    return {
        "pid": pid,
        "start_identity": " ".join(fields[:5]),
        "state": fields[5],
        "argv": shlex.split(fields[6]),
    }


def verify_coordinator(identity: dict, output: Path) -> None:
    args = identity["argv"]
    if identity["pid"] in {os.getpid(), os.getppid()}:
        raise ValueError("Refusing to fence this process or its launcher")
    if (
        "--_worker" in args
        or "-m" not in args
        or args[args.index("-m") + 1] != "experiments.run_reference_baselines"
    ):
        raise ValueError("Takeover PID is not the frozen serial coordinator")
    if "--output" not in args:
        raise ValueError("Coordinator has no explicit reference output")
    path = Path(args[args.index("--output") + 1])
    if not path.is_absolute():
        path = ROOT / path
    if path.resolve() != output.resolve():
        raise ValueError("Coordinator belongs to a different output ledger")


def same_process(observed: dict | None, expected: dict) -> bool:
    return (
        observed is not None
        and observed["pid"] == expected["pid"]
        and " ".join(observed["start_identity"].split())
        == " ".join(expected["start_identity"].split())
        and observed.get("argv") == expected.get("argv")
    )


def amendment_event(path: Path, event: str, **details) -> None:
    record = frozen.read_json(path)
    record.setdefault("events", []).append({"time": frozen.utcnow(), "event": event, **details})
    record["updated_utc"] = frozen.utcnow()
    frozen.write_json(path, record)


def fence_serial(pid: int, output: Path, amendment: Path) -> dict:
    expected = process_identity(pid)
    if expected is None:
        raise ValueError("Requested serial coordinator is no longer running")
    verify_coordinator(expected, output)
    amendment_event(amendment, "serial_fence_requested", process=expected)
    stopped, killed = False, False
    try:
        os.kill(pid, signal.SIGSTOP)
        stopped = True
        for _ in range(50):
            observed = process_identity(pid)
            if observed is None or not same_process(observed, expected):
                raise RuntimeError("Coordinator identity changed while fencing")
            if "T" in observed["state"]:
                break
            time.sleep(0.02)
        else:
            raise RuntimeError("Coordinator did not enter the stopped state")
        amendment_event(amendment, "serial_fenced", process=observed)
        # Final identity check while stopped; never signal the process group.
        checked = process_identity(pid)
        if checked is None or not same_process(checked, expected) or "T" not in checked["state"]:
            raise RuntimeError("Stopped coordinator identity could not be reverified")
        os.kill(pid, signal.SIGKILL)
        killed = True
        amendment_event(amendment, "serial_coordinator_stopped_workers_preserved", process=expected)
        return expected
    finally:
        if stopped and not killed and same_process(process_identity(pid), expected):
            os.kill(pid, signal.SIGCONT)
            amendment_event(amendment, "serial_fence_aborted_parent_resumed")


@contextlib.contextmanager
def coordinator_lease(output: Path, amendment: Path, takeover_pid: int | None):
    with (output / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if takeover_pid is None:
                raise RuntimeError(
                    "A coordinator is active; provide its verified --takeover-pid"
                ) from None
            fence_serial(takeover_pid, output, amendment)
            for _ in range(100):
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.05)
            else:
                raise RuntimeError(
                    "Serial process was fenced but coordinator lock remains held"
                ) from None
        amendment_event(amendment, "coordinator_lease_acquired")
        yield


def create_amendment(
    output: Path, run: dict, workers: int, *, retry_audit_guard_errors: bool = False
) -> Path:
    identity = process_identity(os.getpid())
    if identity is None:
        raise RuntimeError("Cannot identify the new coordinator")
    amendment_id = (
        "parallel-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    directory = output / "execution_amendments"
    path = directory / f"{amendment_id}.json"
    code_hash = frozen.file_hash(Path(__file__))
    code = directory / f"coordinator-{code_hash}.py"
    directory.mkdir(parents=True, exist_ok=True)
    if not code.exists():
        code.write_bytes(Path(__file__).read_bytes())
    worker_source = ROOT / "experiments/reference_worker_v2.py"
    worker_hash = frozen.file_hash(worker_source)
    worker_copy = directory / f"worker-{worker_hash}.py"
    if not worker_copy.exists():
        worker_copy.write_bytes(worker_source.read_bytes())
    state = frozen.read_json(output / "reference_state.json")
    frozen.write_json(
        path,
        {
            "id": amendment_id,
            "protocol": PROTOCOL,
            "original_identity_sha256": run["identity_sha256"],
            "original_concurrency": run["workers"],
            "concurrency": workers,
            "coordinator_source_sha256": code_hash,
            "coordinator_source": str(code.relative_to(output)),
            "worker_protocol": WORKER_PROTOCOL,
            "worker_source_sha256": worker_hash,
            "worker_source": str(worker_copy.relative_to(output)),
            "transport_repair": "Rebind the preloaded judging.run_codex alias to the audited account client after hook installation; prompts/models/grading unchanged",
            "retry_audit_guard_errors": retry_audit_guard_errors,
            "coordinator": identity,
            "started_utc": frozen.utcnow(),
            "reason": "User requested faster parallel evaluation; no additional inference budget",
            "scope": "Concurrency and account-audit binding repair; task identities, instructions, graders, model/tool limits and total budgets stay frozen",
            "timing_comparability": "Wall time and latency before/after this amendment are different execution phases",
            "initial_task_attempts": state["task_attempts"],
            "initial_model_calls": state["model_calls"],
            "initial_scored": [
                key for key, row in state["tasks"].items() if row["status"] == "scored"
            ],
            "initial_running": {
                key: row["attempts"][-1]
                for key, row in state["tasks"].items()
                if row["status"] == "running"
            },
            "events": [],
        },
    )
    return path


def elapsed_seconds(attempt: dict) -> float:
    return (datetime.now(UTC) - datetime.fromisoformat(attempt["started_utc"])).total_seconds()


def live_registered_worker(attempt: dict) -> bool:
    registered = attempt.get("worker")
    if not registered:
        return False
    actual = process_identity(registered["pid"])
    if actual is None or "Z" in actual["state"]:
        return False
    if " ".join(actual["start_identity"].split()) != " ".join(registered["start_identity"].split()):
        return False  # PID was reused; never signal it.
    args = actual["argv"]
    if "--_worker" in args and "--_attempt" in args:
        return args[args.index("--_attempt") + 1] == attempt["id"]
    return (
        "--task" in args
        and "--attempt" in args
        and args[args.index("--attempt") + 1] == attempt["id"]
    )


def stop_expired_worker(output: Path, task_id: str, attempt: dict, process) -> None:
    if process is not None:
        frozen._stop_worker(process)
        return
    registered = attempt.get("worker")
    if not registered:
        raise RuntimeError("Cannot enforce inherited deadline without a registered worker")
    actual = process_identity(registered["pid"])
    if actual is None or "Z" in actual["state"]:
        return
    args = actual["argv"]
    module = args[args.index("-m") + 1] if "-m" in args else None
    task_flag, attempt_flag = (
        ("--_worker", "--_attempt")
        if module == "experiments.run_reference_baselines"
        else ("--task", "--attempt")
    )
    if (
        module not in {"experiments.run_reference_baselines", "experiments.reference_worker_v2"}
        or " ".join(actual["start_identity"].split())
        != " ".join(registered["start_identity"].split())
        or task_flag not in args
        or args[args.index(task_flag) + 1] != task_id
        or attempt_flag not in args
        or args[args.index(attempt_flag) + 1] != attempt["id"]
        or "--output" not in args
        or Path(args[args.index("--output") + 1]).resolve() != output.resolve()
        or os.getpgid(actual["pid"]) != actual["pid"]
    ):
        raise RuntimeError("Cannot verify inherited worker identity/group for deadline enforcement")
    pid = actual["pid"]
    if not same_process(process_identity(pid), actual):
        raise RuntimeError("Worker identity changed before deadline enforcement")
    os.killpg(pid, signal.SIGTERM)
    for _ in range(100):
        current = process_identity(pid)
        if current is None or "Z" in current["state"]:
            return
        if not same_process(current, actual):
            raise RuntimeError("Worker PID changed during bounded shutdown")
        time.sleep(0.1)
    if same_process(process_identity(pid), actual) and os.getpgid(pid) == pid:
        os.killpg(pid, signal.SIGKILL)
        for _ in range(100):
            current = process_identity(pid)
            if current is None or "Z" in current["state"]:
                return
            time.sleep(0.1)
    raise RuntimeError("Inherited worker did not stop at its original deadline")


def completed_before_deadline(output: Path, run: dict, task_id: str, attempt: dict) -> bool:
    path = output / attempt["output"] / "output.json"
    if not path.is_file():
        return False
    try:
        result = frozen.read_json(path)
        started = datetime.fromisoformat(attempt["started_utc"])
        completed = datetime.fromisoformat(result["completed_utc"])
        elapsed = (completed - started).total_seconds()
        return (
            result.get("status") == "scored"
            and result.get("identity_sha256") == run["identity_sha256"]
            and result.get("attempt_id") == attempt["id"]
            and result.get("task") == next(task for task in run["tasks"] if task["id"] == task_id)
            and all(
                type(result.get(key)) in {int, float} and math.isfinite(result[key])
                for key in ("score", "native_score")
            )
            and 0 <= elapsed <= run["worker_timeout_seconds"]
        )
    except (ValueError, TypeError, KeyError):
        return False


def reconcile_task(output: Path, run: dict, task_id: str, owned: dict[str, Any]) -> str:
    row = frozen.read_json(output / "reference_state.json")["tasks"][task_id]
    if row["status"] != "running":
        return "settled"
    attempt = row["attempts"][-1]
    destination = output / attempt["output"]
    destination.mkdir(parents=True, exist_ok=True)
    expired = elapsed_seconds(attempt) > run["worker_timeout_seconds"]
    timely_result = completed_before_deadline(output, run, task_id, attempt)
    if expired and not timely_result:
        stop_expired_worker(output, task_id, attempt, owned.get(task_id))
    with (destination / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if expired:
                if timely_result:
                    stop_expired_worker(output, task_id, attempt, owned.get(task_id))
                    return "active"
                raise RuntimeError(f"Timed-out worker lease remains active for {task_id}") from None
            return "active"
        # Retain this lease through finalization: no late worker can reopen it
        # between the result check and the ledger transition.
        state = frozen.read_json(output / "reference_state.json")
        calls = [call for call in state["calls"].values() if call["attempt_id"] == attempt["id"]]
        process = owned.get(task_id)
        live = process.poll() is None if process is not None else live_registered_worker(attempt)
        result = destination / "output.json"
        if live and not result.exists():
            return "active"
        if not attempt.get("worker") and process is None and not result.exists():
            # A worker may have been forked just before fencing but not registered.
            # Count this attempt against the pool; never launch a replacement.
            if elapsed_seconds(attempt) <= 60:
                return "active"
            raise RuntimeError(
                f"Unregistered inherited attempt {task_id}; process reconciliation required"
            )
        if any(call["status"] == "running" for call in calls):
            raise RuntimeError(
                f"Unsettled model call for {task_id}; no replacement attempt will start"
            )
        if expired and not completed_before_deadline(output, run, task_id, attempt):
            frozen.finish_attempt(
                output,
                run,
                task_id,
                attempt,
                "Worker exceeded its original wall-clock deadline; no budget reset on adoption",
            )
        elif result.exists() and frozen.read_json(result).get("status") == "scored":
            frozen.audit_attempt(output, attempt["id"], run["baseline_id"])
            frozen.finish_attempt(output, run, task_id, attempt)
        else:
            detail = (
                (destination / "stderr.txt").read_text()[-2000:]
                if (destination / "stderr.txt").exists()
                else "No worker stderr captured"
            )
            frozen.finish_attempt(
                output, run, task_id, attempt, "Worker exited without a scored result: " + detail
            )
        return "settled"


def known_audit_guard_failure(row: dict) -> bool:
    return (
        row["status"] == "error"
        and bool(row["attempts"])
        and "Uncharged Codex command blocked" in (row.get("error") or "")
        and row["attempts"][-1].get("worker_transport") != WORKER_PROTOCOL
    )


def launch_task(
    output: Path, run: dict, task_id: str, amendment: Path, *, retry_error: bool = False
):
    from moevo.codex.client import account_environment

    frozen.verify_run(output)
    preceding = frozen.read_json(output / "reference_state.json")["tasks"][task_id]
    if retry_error and not known_audit_guard_failure(preceding):
        raise ValueError("Only the named pre-repair account-audit guard failure may be retried")
    attempt = frozen.charge_task(output, run, task_id, retry_errors=retry_error)
    if attempt is None:
        return None
    destination = output / attempt["output"]
    destination.mkdir(parents=True)
    with frozen.state_transaction(output) as state:
        current = state["tasks"][task_id]["attempts"][-1]
        if current["id"] != attempt["id"]:
            raise ValueError("Parallel worker attempt claim changed")
        current.update(
            worker_transport=WORKER_PROTOCOL, execution_amendment=str(amendment.relative_to(output))
        )
        if retry_error:
            current["replaces_transport_failed_attempt"] = preceding["attempts"][-1]["id"]
    task = next(task for task in run["tasks"] if task["id"] == task_id)
    python = (
        str(ROOT / "data/external/tau3_bench/.venv/bin/python")
        if task["benchmark"] == "tau3_bench"
        else sys.executable
    )
    command = [
        python,
        "-m",
        "experiments.reference_worker_v2",
        "--output",
        str(output),
        "--task",
        task_id,
        "--attempt",
        attempt["id"],
        "--amendment",
        str(amendment),
    ]
    try:
        with (
            (destination / "stdout.txt").open("w") as stdout,
            (destination / "stderr.txt").open("w") as stderr,
        ):
            return subprocess.Popen(
                command,
                cwd=ROOT,
                env=account_environment(),
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
    except Exception as exc:
        frozen.finish_attempt(
            output, run, task_id, attempt, f"Worker launch failed: {type(exc).__name__}: {exc}"
        )
        return None


def publish_progress(output: Path, run: dict, amendment: Path, workers: int, status: str) -> dict:
    record = frozen.read_json(amendment)
    with frozen.state_transaction(output) as state:
        active = [key for key, row in state["tasks"].items() if row["status"] == "running"]
        statuses = [row["status"] for row in state["tasks"].values()]
        state.update(
            status=status,
            phase="parallel" if active else "idle",
            active_tasks=active,
            scored_tasks=statuses.count("scored"),
            error_tasks=sum(value in {"error", "interrupted"} for value in statuses),
            pending_tasks=statuses.count("pending"),
        )
        state["execution_amendment"] = {
            "id": record["id"],
            "path": str(amendment.relative_to(output)),
            "concurrency": workers,
            "original_concurrency": run["workers"],
            "coordinator": record["coordinator"],
            "started_utc": record["started_utc"],
        }
    # Update only fields explicitly excluded from the original identity hash.
    run.update(status=status, updated_utc=frozen.utcnow())
    frozen.write_json(output / "run.json", run)
    return frozen.read_json(output / "reference_state.json")


def coordinate(
    output: Path,
    run: dict,
    amendment: Path,
    workers: int,
    *,
    poll_seconds: float = 1.0,
    retry_audit_guard_errors: bool = False,
) -> dict:
    from moevo.codex.client import require_chatgpt_login

    if require_chatgpt_login() != run["cli_version"]:
        raise ValueError("Account CLI version changed before concurrency amendment")
    owned: dict[str, Any] = {}
    drain = False

    def request_drain(signum, frame):
        nonlocal drain
        drain = True

    previous = {sig: signal.signal(sig, request_drain) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        state = publish_progress(output, run, amendment, workers, "running")
        amendment_event(
            amendment,
            "parallel_phase_started",
            adopted={
                key: row["attempts"][-1]["id"]
                for key, row in state["tasks"].items()
                if row["status"] == "running"
            },
        )
        while True:
            state = frozen.read_json(output / "reference_state.json")
            active = [key for key, row in state["tasks"].items() if row["status"] == "running"]
            for task_id in active:
                if reconcile_task(output, run, task_id, owned) == "settled":
                    process = owned.pop(task_id, None)
                    if process is not None:
                        process.wait(timeout=5)
                    amendment_event(amendment, "task_attempt_settled", task=task_id)
            state = frozen.read_json(output / "reference_state.json")
            active = [key for key, row in state["tasks"].items() if row["status"] == "running"]
            pending = [
                task["id"]
                for task in run["tasks"]
                if state["tasks"][task["id"]]["status"] == "pending"
                or (
                    retry_audit_guard_errors
                    and known_audit_guard_failure(state["tasks"][task["id"]])
                )
            ]
            if not drain:
                for task_id in pending[: max(0, workers - len(active))]:
                    process = launch_task(
                        output,
                        run,
                        task_id,
                        amendment,
                        retry_error=state["tasks"][task_id]["status"] == "error",
                    )
                    if process is not None:
                        owned[task_id] = process
                        amendment_event(amendment, "worker_launched", task=task_id, pid=process.pid)
            state = publish_progress(
                output, run, amendment, workers, "draining" if drain else "running"
            )
            if not state["active_tasks"] and (drain or not state["pending_tasks"]):
                status = (
                    "partial"
                    if state["pending_tasks"]
                    else ("completed" if state["error_tasks"] == 0 else "completed_with_errors")
                )
                amendment_event(
                    amendment,
                    "parallel_phase_finished",
                    status=status,
                    task_attempts=state["task_attempts"],
                    model_calls=state["model_calls"],
                )
                return publish_progress(output, run, amendment, workers, status)
            time.sleep(poll_seconds)
    except Exception as exc:
        amendment_event(
            amendment, "coordinator_blocked_workers_preserved", error=f"{type(exc).__name__}: {exc}"
        )
        publish_progress(output, run, amendment, workers, "blocked")
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--takeover-pid", type=int)
    parser.add_argument(
        "--retry-audit-guard-errors",
        action="store_true",
        help="Charge one replacement attempt only for the identified pre-repair audit-binding failure",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Bounded parallelism requires 1–4 workers per arm")
    output = args.output.resolve()
    run = frozen.verify_run(output)
    with (output / ".handoff.lock").open("a") as handoff:
        try:
            fcntl.flock(handoff, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another coordinator handoff is in progress")
        amendment = create_amendment(
            output, run, args.workers, retry_audit_guard_errors=args.retry_audit_guard_errors
        )
        with coordinator_lease(output, amendment, args.takeover_pid):
            state = coordinate(
                output,
                run,
                amendment,
                args.workers,
                retry_audit_guard_errors=args.retry_audit_guard_errors,
            )
    print(
        json.dumps(
            {
                "status": state["status"],
                "scored": state["scored_tasks"],
                "errors": state["error_tasks"],
                "workers": args.workers,
            }
        )
    )


if __name__ == "__main__":
    main()
