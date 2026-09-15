"""Evaluate the exact development panel with Codex CLI task-only instructions.

The task-only arm and original shared-seed arm use the same benchmark MCP bridge,
task prompts, tools, graders and account transport. They are not two different
model architectures. An isolated worker removes the frozen MOEvo prefix only in
task-only solver calls. Judges and the tau customer retain byte-identical prompts.
Both arms allow successful direct answers with zero tool calls to reach grading.

Prepare without inference, then smoke-test one task and resume the whole panel::

    python -m experiments.run_reference_baselines --source-run SOURCE --output OUT --prepare-only
    python -m experiments.run_reference_baselines --output OUT --resume --task finqa
    python -m experiments.run_reference_baselines --output OUT --resume

No frozen adapter files are modified. Transport errors remain unavailable; native
task failures retain their scores. Attempts and model calls are charged before
execution, including interrupted attempts. Retries require --retry-errors.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import difflib
import fcntl
import hashlib
import inspect
import json
import math
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE_ID = "codex_cli_task_only"
BASELINE_LABEL = "Codex CLI task-only (Astra)"
ARMS = {BASELINE_ID: BASELINE_LABEL, "seed_reference": "Unevolved shared seed (Astra)"}
SCORING_PROTOCOL = "native-grading-with-optional-tool-use-v2"
IMAGE_TAGS = (
    "docker.io/library/moevo-or-runtime:20260915",
    "docker.io/library/moevo-doc-runtime:20260915",
    "docker.io/library/moevo-putnam-runtime:20260915",
    "docker.io/library/moevo-terminal-verifier:20260915",
)
BENCHMARKS = {
    "finqa",
    "bizfinbench2",
    "genebench_pro",
    "amo",
    "travelplanner",
    "dsbench",
    "gdpval",
    "harvey_lab",
    "healthbench_professional",
    "putnambench",
    "oragentbench",
    "terminal_bench_2",
    "tau3_bench",
}
CORE = {"finqa", "bizfinbench2", "genebench_pro", "amo", "travelplanner", "dsbench"}
ACTIVE_CALL: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "reference_call", default=None
)


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    """Atomic, durable publication; readers never observe a partially written ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def state_transaction(output: Path):
    with (output / ".state.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_json(output / "reference_state.json")
        yield state
        state["updated_utc"] = utcnow()
        write_json(output / "reference_state.json", state)


def backend(benchmark: str) -> Any:
    # Lazy imports allow the worker to install its audited account transport first.
    from moevo.codex import multitask_core, multitask_extended

    return multitask_core if benchmark in CORE else multitask_extended


def verify_task(task: dict) -> dict:
    check = backend(task["benchmark"]).preflight(task["benchmark"], task["task_id"])
    if not check.get("structural_preflight_passed") or check.get("adapter_ready") is False:
        raise ValueError(f"Exact-task preflight failed: {task['id']}")
    if check.get("content_sha256") != task["content_sha256"]:
        raise ValueError(f"Task content changed: {task['id']}")
    expected = task.get("input_assets_sha256")
    if expected is not None and check.get("input_assets_sha256") != expected:
        raise ValueError(f"Task input files changed: {task['id']}")
    return check


def source_identity() -> dict[str, str]:
    paths = {Path(__file__)}
    for directory in ("moevo/codex", "experiments/runtime"):
        paths.update(p for p in (ROOT / directory).glob("*") if p.is_file())
    paths.update(
        ROOT / p
        for p in (
            "experiments/validate_domains.py",
            "experiments/run_mini_suite.py",
            "experiments/validate_oragentbench.py",
            "moevo/eval/evaluators/gdpval_judge.py",
            "data/raw/finqa/manifest.json",
            "data/raw/finqa/code/evaluate/evaluate.py",
            "data/raw/genebench_pro/reference_grader.py",
            "data/external/bizfinbench2/utils/JsonPaser.py",
            "data/external/bizfinbench2/benchmark_code/BizFinBench.v2/eval_financial_quantitative_computation.py",
            "data/external/amo_bench/utils.py",
            "data/external/amo_bench/grading.py",
            "data/external/dsbench/data_analysis/compute_answer.py",
            "data/external/simple_evals/healthbench_eval.py",
            "data/external/harvey_lab/lab_core/evaluation/prompts/rubric_criterion.txt",
        )
    )
    paths.update((ROOT / "data/external/travelplanner").rglob("*.py"))
    paths.update(
        p for p in (ROOT / "data/raw/travelplanner/processed/database").rglob("*") if p.is_file()
    )
    # These shared native inputs are not included in the older task preflight hash.
    tau = ROOT / "data/external/tau3_bench"
    paths.update(p for p in (tau / "src/tau2").rglob("*.py") if p.is_file())
    paths.update(p for p in (tau / "data/tau2/domains/retail").glob("*") if p.is_file())
    for relative in ("uv.lock", "pyproject.toml", "source_manifest.json"):
        if (tau / relative).is_file():
            paths.add(tau / relative)
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def native_record_identity(tasks: list[dict]) -> dict[str, str]:
    """Pin complete consumed rows, including gold/rubric fields kept from solvers."""
    result = {}
    for task in tasks:
        module = backend(task["benchmark"])
        if task["benchmark"] in CORE:
            row = module._lookup(task["benchmark"], task["task_id"], ROOT)
        else:
            _, row, _ = module._load(task["benchmark"], task["task_id"], ROOT)
        result[task["id"]] = digest(row)
    return result


def inspect_images() -> dict[str, str]:
    return {
        tag: json.loads(
            subprocess.check_output(["docker", "image", "inspect", tag], text=True, timeout=30)
        )[0]["Id"]
        for tag in IMAGE_TAGS
    }


def _validate_policy(policy: dict) -> None:
    expected = {
        "authentication": "codex_chatgpt_account",
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "judge_model": "gpt-5.6-terra",
        "judge_reasoning_effort": "medium",
    }
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("Reference arm requires account Astra/xhigh and Terra/medium")
    if not isinstance(policy.get("instructions"), str) or not policy["instructions"].strip():
        raise ValueError("Source run must contain the exact nonempty shared instruction prefix")
    for key in ("timeout_seconds", "max_tool_calls"):
        if type(policy.get(key)) is not int or policy[key] <= 0:
            raise ValueError(f"Invalid source policy {key}")


def prepare(
    source_run: Path,
    output: Path,
    *,
    arm: str = BASELINE_ID,
    max_task_attempts: int = 26,
    max_model_calls: int = 512,
) -> dict:
    source_run, output = source_run.resolve(), output.resolve()
    if (
        output == source_run
        or source_run.is_relative_to(output)
        or output.is_relative_to(source_run)
    ):
        raise ValueError("Reference output must be separate from the source run")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory or --resume")
    source = read_json(source_run / "run.json")
    panel = read_json(source_run / "search.json")
    original_protocol = read_json(source_run / "protocol.json")
    if panel.get("split") != "search" or panel.get("protocol") != digest(original_protocol):
        raise ValueError("Only the intact frozen development search panel is allowed")
    tasks = panel["tasks"]
    if (
        len(tasks) != 13
        or {t["benchmark"] for t in tasks} != BENCHMARKS
        or len({t["id"] for t in tasks}) != 13
    ):
        raise ValueError("Expected the exact thirteen-benchmark source panel")
    if tasks != original_protocol["tasks"]:
        raise ValueError("Source panel task identities disagree with its protocol")
    policy = source["policy"]
    if policy != original_protocol["policy"] or any(
        task["seed"] != policy["seed"] for task in tasks
    ):
        raise ValueError("Source policy or task seed differs from the frozen protocol")
    _validate_policy(policy)
    if arm not in ARMS:
        raise ValueError("Unknown reference arm")
    if max_task_attempts < 13 or max_model_calls < 1:
        raise ValueError("Attempt cap must cover 13 tasks; model-call cap must be positive")
    # Launcher changes do not alter task protocols, but frozen solver/grader changes do.
    changed = [
        p
        for p, sha in original_protocol["sources"].items()
        if p.startswith("moevo/codex/") and file_hash(ROOT / p) != sha
    ]
    if changed:
        raise ValueError(f"Source solver/grader code changed: {changed}")
    checks = {task["id"]: verify_task(task) for task in tasks}
    images = inspect_images()
    if images[IMAGE_TAGS[0]] != source.get("runtime_image"):
        raise ValueError("Base runtime differs from the source run's immutable image")
    identity = {
        "version": 1,
        "baseline_id": arm,
        "baseline_label": ARMS[arm],
        "source_run": str(source_run),
        "source_panel_sha256": file_hash(source_run / "search.json"),
        "source_protocol_sha256": file_hash(source_run / "protocol.json"),
        "source_policy_sha256": digest(policy),
        "native_record_sha256": native_record_identity(tasks),
        "tasks": tasks,
        "frozen_panel_digest": digest(panel),
        "frozen_source_protocol_digest": digest(original_protocol),
        "policy": {**policy, "instructions": "" if arm == BASELINE_ID else policy["instructions"]},
        "adapter_policy": policy,
        "effective_shared_instructions": "" if arm == BASELINE_ID else policy["instructions"],
        "instruction_removal": "Exact source prefix, solver role only, once per solver call; other prompts byte-identical"
        if arm == BASELINE_ID
        else "None: original unevolved shared seed instructions preserved",
        "scoring_protocol": SCORING_PROTOCOL,
        "source_scoring_protocol_difference": "Zero tool calls no longer exclude completed responses; native graders determine task success. Original source-run gate errors remain unchanged and are not matched scored comparisons.",
        "comparison_scope": "Matched shared-instruction ablation using Codex CLI and the benchmark MCP bridge; not an unmodified commercial CLI or a separate bare-model architecture",
        "sources": source_identity(),
        "runtime_images": images,
        "max_task_attempts": max_task_attempts,
        "max_model_calls": max_model_calls,
        "workers": 1,
        "worker_timeout_seconds": max(5400, policy["timeout_seconds"] + 180),
    }
    run = {
        **identity,
        "identity_sha256": digest(identity),
        "status": "prepared",
        "created_utc": utcnow(),
        "updated_utc": utcnow(),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", run)
    write_json(output / "source_panel.json", panel)
    write_json(output / "source_protocol.json", original_protocol)
    for task_id, check in checks.items():
        write_json(output / "preflight" / f"{task_id}.json", check)
    write_json(
        output / "reference_state.json",
        {
            "version": 1,
            "baseline_id": arm,
            "identity_sha256": run["identity_sha256"],
            "status": "prepared",
            "phase": "waiting",
            "task_attempts": 0,
            "model_calls": 0,
            "scored_tasks": 0,
            "error_tasks": 0,
            "pending_tasks": len(tasks),
            "model_usage_complete": True,
            "model_usage": {},
            "calls": {},
            "tasks": {
                task["id"]: {
                    **task,
                    "status": "pending",
                    "score": None,
                    "native_score": None,
                    "attempts": [],
                    "output": None,
                    "error": None,
                }
                for task in tasks
            },
            "updated_utc": utcnow(),
        },
    )
    return run


def verify_run(output: Path) -> dict:
    run = read_json(output / "run.json")
    identity = {
        k: v
        for k, v in run.items()
        if k
        not in {"identity_sha256", "status", "created_utc", "updated_utc", "cli_version", "error"}
    }
    if digest(identity) != run["identity_sha256"]:
        raise ValueError("Reference protocol identity changed")
    state = read_json(output / "reference_state.json")
    if state["identity_sha256"] != run["identity_sha256"]:
        raise ValueError("Reference ledger belongs to another protocol")
    changed = [
        p
        for p, sha in run["sources"].items()
        if not (ROOT / p).is_file() or file_hash(ROOT / p) != sha
    ]
    if changed:
        raise ValueError(f"Frozen reference code/assets changed: {changed}")
    if native_record_identity(run["tasks"]) != run["native_record_sha256"]:
        raise ValueError("Frozen native task/gold record changed")
    for image_id in run["runtime_images"].values():
        actual = json.loads(
            subprocess.check_output(["docker", "image", "inspect", image_id], text=True, timeout=30)
        )[0]["Id"]
        if actual != image_id:
            raise ValueError("Frozen runtime image is unavailable")
    for name, key in (
        ("source_panel.json", "frozen_panel_digest"),
        ("source_protocol.json", "frozen_source_protocol_digest"),
    ):
        if digest(read_json(output / name)) != run[key]:
            raise ValueError("Frozen source-panel/protocol copy changed")
    return run


def transformed_prompt(
    prompt: str, prefix: str, role: str, model: str, effort: str, arm: str = BASELINE_ID
) -> tuple[str, dict]:
    if role not in {"solver", "tau_agent", "tau_customer", "judge"}:
        raise ValueError("Unknown model-call role; refusing unaudited inference")
    required = ("gpt-5.6-terra", "medium") if role == "judge" else ("gpt-6-astra", "xhigh")
    if (model, effort) != required:
        raise ValueError(f"Unexpected model/settings for {role}")
    if arm not in ARMS:
        raise ValueError("Unknown reference arm")
    solver = role in {"solver", "tau_agent"}
    remove = solver and arm == BASELINE_ID
    effective = prompt
    if solver and not prompt.startswith(prefix + "\n"):
        raise ValueError("Solver did not receive the exact frozen instruction prefix")
    if remove:
        # Adapters append one or two delimiter newlines. Strip only that separator.
        suffix = prompt[len(prefix) :]
        separator = "\n\n" if suffix.startswith("\n\n") else "\n"
        effective = suffix[len(separator) :]
        if not effective:
            raise ValueError("Removing shared instructions left no benchmark prompt")
    elif not solver and prompt.startswith(prefix + "\n"):
        raise ValueError("Unexpected shared instructions in judge/customer prompt")
    return effective, {
        "role": role,
        "model": model,
        "effort": effort,
        "prefix_removed": remove,
        "removal_count": int(remove),
        "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        "original_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "effective_prompt_sha256": hashlib.sha256(effective.encode()).hexdigest(),
        "unchanged": prompt == effective,
    }


def caller_role(frame) -> tuple[str, dict]:
    module, function = frame.f_globals.get("__name__"), frame.f_code.co_name
    role = {
        ("moevo.codex.container_runtime", "solve_in_container"): "solver",
        ("moevo.codex.container_runtime_v2", "solve_in_container"): "solver",
        ("moevo.codex.multitask_extended", "_health"): "solver",
        ("moevo.codex.judging", "run_judge"): "judge",
    }.get((module, function))
    if module == "moevo.codex.tau_pilot" and function == "generate":
        role = {"agent_response": "tau_agent", "user_simulator_response": "tau_customer"}.get(
            frame.f_locals.get("call_name")
        )
    if role is None:
        raise ValueError(f"Unrecognized account call site {module}.{function}")
    return role, {
        "module": module,
        "function": function,
        "native_call_name": frame.f_locals.get("call_name"),
    }


def charge_model_call(output: Path, run: dict, task_id: str, attempt_id: str, proof: dict) -> str:
    with state_transaction(output) as state:
        row = state["tasks"][task_id]
        if row["status"] != "running" or row["attempts"][-1]["id"] != attempt_id:
            raise ValueError("Model call does not belong to the active charged task attempt")
        if state["model_calls"] >= run["max_model_calls"]:
            raise RuntimeError("Reference model-call budget exhausted before invocation")
        call_id = f"call-{state['model_calls']:05d}"
        state["model_calls"] += 1
        state["model_usage_complete"] = False
        state["calls"][call_id] = {
            **proof,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "status": "running",
            "started_utc": utcnow(),
            "usage": None,
        }
    return call_id


def _docker_argument_ids(args, images: dict):
    if not isinstance(args, (list, tuple)) or not args or str(args[0]) != "docker":
        return args
    return [images.get(str(part), part) for part in args]


def install_worker_hooks(output: Path, run: dict, task_id: str, attempt_id: str) -> None:
    """Process-local audit/removal hooks; never patch files or another run's process."""
    from moevo.codex import client

    prefix = run["adapter_policy"]["instructions"]
    original_call, original_command = client.run_codex, client.build_command

    def audited_command(*args, **kwargs):
        command = original_command(*args, **kwargs)
        call = ACTIVE_CALL.get()
        if call is None:
            raise ValueError("Uncharged Codex command blocked")
        if (
            'forced_login_method="chatgpt"' not in command
            or "--ignore-user-config" not in command
            or "features.shell_tool=false" not in command
            or "-m" not in command
            or command[command.index("-m") + 1] != call["model"]
        ):
            raise ValueError("Codex command departed from account-only frozen transport")
        environment = client.account_environment()
        if any(
            k.endswith("API_KEY") or k in {"OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"}
            for k in environment
        ):
            raise ValueError("API transport environment was not removed")
        write_json(
            call["directory"] / "command.json",
            {"argv": command, "forced_chatgpt_login": True, "api_credentials_removed": True},
        )
        return command

    def audited_call(prompt, **kwargs):
        current = inspect.currentframe()
        assert current is not None and current.f_back is not None
        role, caller = caller_role(current.f_back)
        model, effort = kwargs.get("model", client.DEFAULT_MODEL), kwargs.get("effort", "xhigh")
        effective, proof = transformed_prompt(
            prompt, prefix, role, model, effort, run["baseline_id"]
        )
        proof["caller"] = caller
        call_id = charge_model_call(output, run, task_id, attempt_id, proof)
        directory = output / "model_calls" / call_id
        directory.mkdir(parents=True)
        (directory / "prompt-original.txt").write_text(prompt)
        (directory / "prompt-effective.txt").write_text(effective)
        (directory / "prompt.diff").write_text(
            "".join(
                difflib.unified_diff(
                    prompt.splitlines(keepends=True),
                    effective.splitlines(keepends=True),
                    fromfile="original",
                    tofile="task-only",
                )
            )
        )
        write_json(directory / "proof.json", proof)
        active = {"directory": directory, "model": model, "processes": [], "call_id": call_id}
        token = ACTIVE_CALL.set(active)
        try:
            response = original_call(effective, **kwargs)
        except BaseException as exc:
            with state_transaction(output) as state:
                state["calls"][call_id].update(
                    status="error", error=f"{type(exc).__name__}: {exc}", ended_utc=utcnow()
                )
                state["model_usage_complete"] = False
            raise
        else:
            with state_transaction(output) as state:
                state["calls"][call_id].update(
                    status="completed",
                    usage=response.usage,
                    duration_s=response.duration_s,
                    ended_utc=utcnow(),
                )
                for key, value in response.usage.items():
                    if type(value) is int:
                        state["model_usage"][key] = state["model_usage"].get(key, 0) + value
                state["model_usage_complete"] = all(
                    row["status"] == "completed" for row in state["calls"].values()
                )
            return response
        finally:
            for process in active["processes"]:
                _stop_worker(process)
            ACTIVE_CALL.reset(token)

    client.build_command, client.run_codex = audited_command, audited_call
    # Some verifier helpers embed image tags. Replace only exact Docker argv
    # elements, retaining the same commands and immutable preselected images.
    original_popen = subprocess.Popen

    def pinned_popen(args, *positional, **kwargs):
        process = original_popen(
            _docker_argument_ids(args, run["runtime_images"]), *positional, **kwargs
        )
        active = ACTIVE_CALL.get()
        if active is not None and kwargs.get("start_new_session"):
            active["processes"].append(process)
            with state_transaction(output) as state:
                state["calls"][active["call_id"]]["process"] = {
                    "pid": process.pid,
                    "argv_sha256": digest(list(args)),
                    "started_utc": utcnow(),
                    "start_identity": process_start_identity(process.pid),
                }
        return process

    subprocess.Popen = pinned_popen  # type: ignore[assignment]


def score_report(report: dict, task: dict) -> tuple[float, float]:
    if report.get("task_id") != task["task_id"] or report.get("e2e_validated") is not True:
        raise ValueError("Reference adapter did not validate the exact requested task")
    native = report.get("score")
    value = (
        report.get("criterion_pass_rate", native) if task["benchmark"] == "harvey_lab" else native
    )
    if (
        isinstance(native, bool)
        or not isinstance(native, (int, float))
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(native)
        or not math.isfinite(value)
    ):
        raise ValueError("Reference report has no finite native/search score")
    return min(1.0, max(0.0, float(value))), float(native)


def audit_attempt(output: Path, attempt_id: str, arm: str = BASELINE_ID) -> dict:
    calls = {
        key: row
        for key, row in read_json(output / "reference_state.json")["calls"].items()
        if row["attempt_id"] == attempt_id
    }
    solvers = [row for row in calls.values() if row["role"] in {"solver", "tau_agent"}]
    remove = arm == BASELINE_ID
    if not solvers or not all(
        row["prefix_removed"] == remove
        and row["removal_count"] == int(remove)
        and (remove or row["unchanged"])
        for row in solvers
    ):
        raise ValueError("Solver instruction handling does not match the frozen arm")
    if any(
        not row["unchanged"] for row in calls.values() if row["role"] in {"judge", "tau_customer"}
    ):
        raise ValueError("A judge or customer prompt changed")
    if any(row["status"] != "completed" for row in calls.values()):
        raise ValueError("The attempted native result contains an incomplete model call")
    for call_id in calls:
        if not (output / "model_calls" / call_id / "command.json").is_file():
            raise ValueError("Missing actual Codex command proof")
    return {
        "call_ids": list(calls),
        "solver_calls": len(solvers),
        "prefix_removals": len(solvers) if remove else 0,
        "judge_calls": sum(row["role"] == "judge" for row in calls.values()),
        "customer_calls": sum(row["role"] == "tau_customer" for row in calls.values()),
        "judge_customer_prompts_unchanged": True,
    }


def process_start_identity(pid: int) -> str:
    return subprocess.check_output(
        ["ps", "-p", str(pid), "-o", "lstart="], text=True, timeout=5
    ).strip()


def worker(output: Path, task_id: str, attempt_id: str) -> None:
    run = read_json(output / "run.json")
    row = read_json(output / "reference_state.json")["tasks"][task_id]
    if row["status"] != "running" or row["attempts"][-1]["id"] != attempt_id:
        raise ValueError("Worker lacks an active precharged task attempt")
    destination = output / row["attempts"][-1]["output"]
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {
            "pid": os.getpid(),
            "start_identity": process_start_identity(os.getpid()),
            "started_utc": utcnow(),
        }
        with state_transaction(output) as state:
            current = state["tasks"][task_id]
            if current["status"] != "running" or current["attempts"][-1]["id"] != attempt_id:
                raise ValueError("Worker attempt was reconciled before startup")
            current["attempts"][-1]["worker"] = identity
        _worker_evaluate(output, run, task_id, attempt_id)


def _worker_evaluate(output: Path, run: dict, task_id: str, attempt_id: str) -> None:
    install_worker_hooks(output, run, task_id, attempt_id)

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Reference worker received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    from moevo.codex import container_runtime_v2, domain_tasks

    domain_tasks.solve_in_container = container_runtime_v2.solve_in_container
    task = next(t for t in run["tasks"] if t["id"] == task_id)
    row = read_json(output / "reference_state.json")["tasks"][task_id]
    if row["status"] != "running" or row["attempts"][-1]["id"] != attempt_id:
        raise ValueError("Worker lacks an active precharged task attempt")
    destination = output / row["attempts"][-1]["output"]
    verify_task(task)
    if native_record_identity([task])[task_id] != run["native_record_sha256"][task_id]:
        raise ValueError("Native task/gold changed before worker execution")
    module = backend(task["benchmark"])
    if task["benchmark"] == "tau3_bench":
        # The coordinator already selected upstream Python. Keep its original
        # worker in this isolated process so agent/user calls retain our audit.
        def tau_in_process(policy, attempt_output, image, native_id):
            module._tau_worker(native_id, attempt_output, policy)
            return read_json(attempt_output / "native-report.json")

        module.HANDLERS = {**module.HANDLERS, "tau3_bench": tau_in_process}
    report = module.evaluate(
        task["benchmark"],
        task["task_id"],
        run["adapter_policy"],
        destination / "artifacts",
        run["runtime_images"][IMAGE_TAGS[0]],
    )
    score, native = score_report(report, task)
    audit = audit_attempt(output, attempt_id, run["baseline_id"])
    write_json(
        destination / "output.json",
        {
            "baseline_id": run["baseline_id"],
            "baseline_label": run["baseline_label"],
            "identity_sha256": run["identity_sha256"],
            "attempt_id": attempt_id,
            "task": task,
            "status": "scored",
            "score": score,
            "native_score": native,
            "native_metrics": {"score": native, "grade": report["grade"]},
            "protocol": report.get("protocol"),
            "scoring_protocol": SCORING_PROTOCOL,
            "usage": report.get("usage", {}),
            "prompt_audit": audit,
            "completed_utc": utcnow(),
        },
    )


def charge_task(
    output: Path, run: dict, task_id: str, *, retry_errors: bool = False
) -> dict | None:
    with state_transaction(output) as state:
        row = state["tasks"][task_id]
        if row["status"] == "scored":
            return None
        if row["status"] in {"error", "interrupted"} and not retry_errors:
            return None
        if row["status"] == "running":
            raise ValueError("Active attempt must be reconciled before retry")
        if state["task_attempts"] >= run["max_task_attempts"]:
            raise RuntimeError("Reference task-attempt budget exhausted before invocation")
        state["task_attempts"] += 1
        attempt = {
            "id": f"attempt-{state['task_attempts']:04d}",
            "status": "running",
            "output": f"tasks/{task_id}/attempt-{state['task_attempts']:04d}",
            "error": None,
            "started_utc": utcnow(),
            "ended_utc": None,
        }
        row["attempts"].append(attempt)
        row.update(status="running", score=None, native_score=None, output=None, error=None)
        state.update(status="running", phase=f"evaluating:{task_id}")
    return attempt


def finish_attempt(
    output: Path,
    run: dict,
    task_id: str,
    attempt: dict,
    error: str | None = None,
    *,
    interrupted: bool = False,
) -> None:
    destination = output / attempt["output"]
    result_path = destination / "output.json"
    result = read_json(result_path) if error is None and result_path.exists() else None
    if result is not None and (
        result.get("identity_sha256") != run["identity_sha256"]
        or result.get("attempt_id") != attempt["id"]
        or result.get("task") != next(t for t in run["tasks"] if t["id"] == task_id)
    ):
        result, error = None, "Worker result identity mismatch"
    if result is not None and (
        result.get("status") != "scored"
        or type(result.get("score")) not in {int, float}
        or not math.isfinite(result["score"])
    ):
        result, error = None, "Malformed worker score"
    if result is None:
        error = error or "Worker exited without an audited result"
        failure = {
            "baseline_id": run["baseline_id"],
            "identity_sha256": run["identity_sha256"],
            "task": next(t for t in run["tasks"] if t["id"] == task_id),
            "attempt_id": attempt["id"],
            "status": "interrupted" if interrupted else "error",
            "score": None,
            "native_score": None,
            "error": error,
            "ended_utc": utcnow(),
        }
        # Never replace a previously completed task-level result.
        write_json(destination / "error.json", failure)
        if not result_path.exists():
            write_json(result_path, failure)
    with state_transaction(output) as state:
        row = state["tasks"][task_id]
        if row["attempts"][-1]["id"] != attempt["id"]:
            raise ValueError("Attempt completion identity mismatch")
        status = "scored" if result is not None else ("interrupted" if interrupted else "error")
        row.update(
            status=status,
            score=result["score"] if result else None,
            native_score=result["native_score"] if result else None,
            output=str(result_path.relative_to(output)),
            error=error,
        )
        row["attempts"][-1].update(status=status, error=error, ended_utc=utcnow())
        statuses = [task["status"] for task in state["tasks"].values()]
        state.update(
            scored_tasks=statuses.count("scored"),
            error_tasks=sum(value in {"error", "interrupted"} for value in statuses),
            pending_tasks=statuses.count("pending"),
        )


def recover_interrupted(output: Path, run: dict) -> None:
    state = read_json(output / "reference_state.json")
    # Hard coordinator loss does not imply a separately-sessioned worker/CLI died.
    # Refuse ambiguous recovery instead of starting an overlapping charged retry.
    with contextlib.ExitStack() as leases:
        for row in state["tasks"].values():
            if row["status"] != "running":
                continue
            destination = output / row["attempts"][-1]["output"]
            destination.mkdir(parents=True, exist_ok=True)
            lock = leases.enter_context((destination / ".worker.lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "A previous reference worker is still active; wait for its completion before resuming"
                ) from exc
        # Keep leases until reconciliation finishes. A worker that had not started
        # yet cannot register or begin inference between the check and the update.
        state = read_json(output / "reference_state.json")
        if any(call["status"] == "running" for call in state["calls"].values()):
            raise RuntimeError(
                "Unsettled model call after interruption; process and usage reconciliation is required before retry"
            )
        for task_id, row in state["tasks"].items():
            if row["status"] == "running":
                attempt = row["attempts"][-1]
                # A committed result survives coordinator interruption and is not rerun.
                finish_attempt(
                    output,
                    run,
                    task_id,
                    attempt,
                    interrupted=not (output / attempt["output"] / "output.json").exists(),
                )


def _stop_worker(process) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def execute(output: Path, run: dict, task_ids: list[str], *, retry_errors: bool = False) -> dict:
    from moevo.codex.client import account_environment, require_chatgpt_login

    version = require_chatgpt_login()
    if run.get("cli_version") not in {None, version}:
        raise ValueError("Codex CLI version changed since reference execution")
    run.update(status="running", cli_version=version, updated_utc=utcnow())
    write_json(output / "run.json", run)
    recover_interrupted(output, run)
    for task_id in task_ids:
        verify_run(output)
        attempt = charge_task(output, run, task_id, retry_errors=retry_errors)
        if attempt is None:
            continue
        destination = output / attempt["output"]
        destination.mkdir(parents=True)
        task = next(t for t in run["tasks"] if t["id"] == task_id)
        python = (
            str(ROOT / "data/external/tau3_bench/.venv/bin/python")
            if task["benchmark"] == "tau3_bench"
            else sys.executable
        )
        command = [
            python,
            "-m",
            "experiments.run_reference_baselines",
            "--output",
            str(output),
            "--_worker",
            task_id,
            "--_attempt",
            attempt["id"],
        ]
        error, interrupted, process = None, False, None
        with (
            (destination / "stdout.txt").open("w") as stdout,
            (destination / "stderr.txt").open("w") as stderr,
        ):
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=account_environment(),
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                returncode = process.wait(timeout=run["worker_timeout_seconds"])
                if returncode:
                    error = (
                        f"Worker exit {returncode}: "
                        + (destination / "stderr.txt").read_text()[-2000:]
                    )
            except subprocess.TimeoutExpired:
                error = "Reference task worker exceeded its declared wall-clock cap"
            except BaseException as exc:
                error, interrupted = f"{type(exc).__name__}: {exc}", True
            finally:
                if process is not None:
                    _stop_worker(process)
        finish_attempt(output, run, task_id, attempt, error, interrupted=interrupted)
        if interrupted:
            break
    with state_transaction(output) as state:
        statuses = [row["status"] for row in state["tasks"].values()]
        status = (
            "completed"
            if all(value == "scored" for value in statuses)
            else (
                "completed_with_errors"
                if all(value in {"scored", "error", "interrupted"} for value in statuses)
                else "partial"
            )
        )
        state.update(
            status=status,
            phase="idle",
            scored_tasks=statuses.count("scored"),
            error_tasks=sum(value in {"error", "interrupted"} for value in statuses),
            pending_tasks=statuses.count("pending"),
        )
    run.update(status=status, updated_utc=utcnow())
    write_json(output / "run.json", run)
    return read_json(output / "reference_state.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--task", action="append", help="Manifest id; omit to execute all thirteen")
    parser.add_argument(
        "--all13", action="store_true", help="Explicit synonym for the default complete panel"
    )
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-task-attempts", type=int)
    parser.add_argument("--max-model-calls", type=int)
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    parser.add_argument("--_attempt", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args._worker:
        if not args._attempt:
            parser.error("Worker attempt id required")
        worker(output, args._worker, args._attempt)
        return
    if args.task and args.all13:
        parser.error("Use --task or --all13")
    if args.resume:
        run = verify_run(output)
        if args.source_run and str(args.source_run.resolve()) != run["source_run"]:
            parser.error("Resume source run differs from the frozen protocol")
        if args.arm is not None and args.arm != run["baseline_id"]:
            parser.error("Resume arm differs from the frozen protocol")
        if any(
            value is not None and value != run[key]
            for value, key in (
                (args.max_task_attempts, "max_task_attempts"),
                (args.max_model_calls, "max_model_calls"),
            )
        ):
            parser.error("Resume budgets differ from the frozen protocol")
    else:
        if not args.source_run:
            parser.error("--source-run is required when preparing a reference run")
        run = prepare(
            args.source_run,
            output,
            arm=args.arm or BASELINE_ID,
            max_task_attempts=args.max_task_attempts if args.max_task_attempts is not None else 26,
            max_model_calls=args.max_model_calls if args.max_model_calls is not None else 512,
        )
    tasks = args.task or [task["id"] for task in run["tasks"]]
    if len(tasks) != len(set(tasks)) or any(
        task not in {t["id"] for t in run["tasks"]} for task in tasks
    ):
        parser.error("Select unique exact manifest task ids")
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": run["status"],
                    "output": str(output),
                    "model_calls": 0,
                    "tasks": len(run["tasks"]),
                }
            )
        )
        return
    with (output / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another reference coordinator is active")
        state = execute(output, run, tasks, retry_errors=args.retry_errors)
    print(
        json.dumps(
            {
                "status": state["status"],
                "scored": state["scored_tasks"],
                "errors": state["error_tasks"],
                "task_attempts": state["task_attempts"],
                "model_calls": state["model_calls"],
            }
        )
    )


if __name__ == "__main__":
    main()
