"""Resumable eight-slice SuperHarness development study; never evaluates final.

Every slice re-scores its incoming harness on its own panel. Improvements are
reported against the original seed on that same panel, and on a separate fixed
selection monitor. Selection observations never enter mutation feedback.

Resume restores completed stages/checkpoints and charges interrupted attempts.
It does not replay an exact pending proposal after interruption: a generated
candidate not committed to a controller checkpoint may be proposed again, using
a new charged call. Proposal artifacts preserve the abandoned candidate.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import fcntl
import gzip
import hashlib
import inspect
import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.instruction_search import (
    PROTOCOL,
    build_instruction_prompt,
    generate_account_proposal,
    parse_candidate_response,
    source_sha256,
    validate_instruction_change,
)
from experiments.prepare_multidomain_study import BENCHMARKS
from moevo.codex.finance_pilot import write_json
from moevo.controller import MoevoController
from moevo.core.config import MoevoConfig
from moevo.generation.schedule import EvaluationBudgetError
from moevo.reporting.aggregate import aggregate_panel

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "experiments/sliced_task_adapter.py"
SLICE_NAMES = [f"S{i}" for i in range(1, 9)]
SETTINGS = {
    "mutation_protocol": PROTOCOL,
    "mutations_per_slice": 4,
    "population_size": 24,
    "num_islands": 2,
    "selection": "nsga3",
    "screen_domains": 3,
    "screen_tasks_per_domain": 1,
    "screen_audit_every": 4,
    "evaluation_concurrency": 2,
    "retry_attempts": 0,
    "maximum_task_attempts": 728,
    "maximum_proposal_calls": 32,
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _source_identity(adapter: Path) -> dict:
    sources = [
        Path(__file__),
        ROOT / "experiments/instruction_search.py",
        adapter,
        ROOT / "moevo/controller.py",
        ROOT / "moevo/reporting/aggregate.py",
    ]
    for directory in (
        "moevo/codex",
        "moevo/core",
        "moevo/generation",
        "moevo/search",
        "experiments/runtime",
    ):
        sources.extend(sorted((ROOT / directory).glob("*.py")))
    sources.extend(
        ROOT / name
        for name in (
            "experiments/validate_domains.py",
            "experiments/pilot_task_adapter.py",
            "experiments/prepare_multidomain_study.py",
            "experiments/instruction_search.py",
            "experiments/run_mini_suite.py",
        )
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(sources))
    }


def load_study(study: Path) -> dict:
    allocation_path = study / "allocation.json"
    allocation = (
        read_json(allocation_path)
        if allocation_path.exists()
        else json.loads(gzip.decompress((study / "allocation.json.gz").read_bytes()))
    )
    return {
        "summary": read_json(study / "summary.json"),
        "allocation": allocation,
        "panels": {
            name: read_json(study / "panels" / f"{name}.json")
            for name in [*SLICE_NAMES, "selection", "final"]
        },
    }


def readiness(
    study: Path, *, through_slice: int = 8, required_benchmarks: list[str] | None = None
) -> dict:
    """Read-only readiness; structural readiness is not successful E2E execution."""
    if not 1 <= through_slice <= 8:
        raise ValueError("through_slice must be between one and eight")
    required = set(BENCHMARKS if required_benchmarks is None else required_benchmarks)
    blockers = []
    try:
        data = load_study(study)
    except (OSError, ValueError) as exc:
        return {"ready": False, "blockers": [str(exc)], "model_calls": 0}
    summary, panels = data["summary"], data["panels"]
    if set(summary.get("benchmarks", [])) != required:
        blockers.append(
            "Study benchmark inventory differs from the required complete benchmark set"
        )
    unhashed = {key: value for key, value in summary.items() if key != "study_sha256"}
    if summary.get("study_sha256") != digest({"summary": unhashed, "manifests": panels}):
        blockers.append("Study summary/panel fingerprint mismatch")
    if summary.get("protocol", {}).get("allocation_sha256") != data["allocation"].get(
        "manifest_sha256"
    ):
        blockers.append("Allocation identity mismatch")
    allocation_payload = {
        key: value for key, value in data["allocation"].items() if key != "manifest_sha256"
    }
    allocation_digest = hashlib.sha256(
        json.dumps(
            allocation_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    if data["allocation"].get("manifest_sha256") != allocation_digest:
        blockers.append("Allocation content fingerprint mismatch")
    policy = summary.get("protocol", {}).get("policy", {})
    for key, expected in {
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "authentication": "codex_chatgpt_account",
        "judge_model": "gpt-5.6-terra",
        "judge_reasoning_effort": "medium",
    }.items():
        if policy.get(key) != expected:
            blockers.append(f"Policy must pin {key}={expected}")
    if not isinstance(policy.get("instructions"), str) or not policy["instructions"].strip():
        blockers.append("Missing original shared instructions")
    selected = [*SLICE_NAMES[:through_slice], "selection"]
    components = {}
    total_tasks = 0
    for name, panel in panels.items():
        tasks = panel.get("tasks", [])
        if not tasks or len({task.get("id") for task in tasks}) != len(tasks):
            blockers.append(f"{name}: empty panel or duplicate evaluation IDs")
        if panel.get("split") != ("search" if name in SLICE_NAMES else name):
            blockers.append(f"{name}: incorrect split role")
        if set(panel.get("confirmation_ids", [])) != {task.get("id") for task in tasks}:
            blockers.append(f"{name}: confirmation panel must contain every task")
        for task in tasks:
            component = task.get("component")
            if not component:
                blockers.append(f"{name}: task has no group-disjoint component identity")
            elif component in components and components[component] != name:
                blockers.append("Task family reused across study partitions")
            components[component] = name
        if name in selected:
            total_tasks += len(tasks)
            if {task.get("benchmark") for task in tasks} != required:
                blockers.append(
                    f"{name}: missing or unexpected benchmarks; no benchmark may be dropped"
                )
            for task in tasks:
                if task.get("adapter_ready") is not True:
                    blockers.append(
                        f"{name}: unsupported {task.get('benchmark')}/{task.get('task_id')}"
                    )
    # Summary flags are independently enforced, so editing one task's readiness
    # flag cannot silently override a preparation blocker.
    for name, missing in summary.get("missing_benchmarks", {}).items():
        if name in selected and missing:
            blockers.append(f"{name}: preparation reported missing benchmarks {missing}")
    for item in summary.get("unsupported_tasks", []):
        if item.get("partition") in selected:
            blockers.append(
                f"{item['partition']}: preparation reported unsupported {item.get('benchmark')}/{item.get('task_id')}"
            )
    expected_cap = (
        sum(5 * len(panels[name]["tasks"]) for name in SLICE_NAMES)
        + sum(len(panels[name]["tasks"]) for name in SLICE_NAMES[1:])
        + 9 * len(panels["selection"]["tasks"])
    )
    if expected_cap > SETTINGS["maximum_task_attempts"]:
        blockers.append(f"Study requires {expected_cap} task attempts; fixed limit is 728")
    if summary.get("task_attempt_caps", {}).get("total") != expected_cap:
        blockers.append("Prepared resource cap does not match the fixed eight-slice protocol")
    return {
        "ready": not blockers,
        "blockers": sorted(set(blockers)),
        "model_calls": 0,
        "through_slice": SLICE_NAMES[through_slice - 1],
        "selected_panel_tasks": total_tasks,
        "maximum_task_attempts": expected_cap,
        "maximum_proposal_calls": 32,
        "final_evaluation_allowed": False,
        "scope": "development study with fixed selection monitor; final never evaluated",
        "future_preparation_blockers": [
            item
            for item in summary.get("unsupported_tasks", [])
            if item.get("partition") not in selected
        ],
    }


@contextlib.contextmanager
def stage_environment(directory: Path):
    before = {key: os.environ.get(key) for key in ("MOEVO_SLICED_DIR", "MOEVO_ACCOUNT_ONLY")}
    os.environ.update(MOEVO_SLICED_DIR=str(directory), MOEVO_ACCOUNT_ONLY="1")
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class PanelLedger:
    """Full-panel scoring with durable pre-call charges; excludes mutation feedback."""

    def __init__(
        self, path: Path, signature: str, tasks: list[dict], cap: int, executor, judge: dict
    ):
        self.path, self.tasks, self.cap, self.executor, self.judge = (
            path,
            tasks,
            cap,
            executor,
            judge,
        )
        self.state = (
            read_json(path)
            if path.exists()
            else {"signature": signature, "task_evaluations": 0, "cache": {}, "events": []}
        )
        if self.state["signature"] != signature:
            raise ValueError("Full-panel ledger configuration changed")
        self.save()

    def save(self):
        write_json(self.path, self.state)

    async def ensure(self, code_path: Path):
        code = code_path.read_text()
        candidate = digest(code)
        missing = [
            task
            for task in self.tasks
            if digest([self.state["signature"], code, task["id"]]) not in self.state["cache"]
        ]
        if self.state["task_evaluations"] + len(missing) > self.cap:
            raise EvaluationBudgetError("Full panel will not fit remaining task-attempt budget")

        async def execute(task):
            key = digest([self.state["signature"], code, task["id"]])
            self.state["task_evaluations"] += 1
            self.save()  # Failed and interrupted attempts remain charged.
            started = time.monotonic()
            try:
                response = await asyncio.to_thread(
                    self.executor, str(code_path), {**task, **self.judge}
                )
                if inspect.isawaitable(response):
                    response = await response
                score = response.get("score") if isinstance(response, dict) else None
                if (
                    not isinstance(response, dict)
                    or response.get("status") != "scored"
                    or isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                ):
                    raise ValueError("Full-panel adapter did not return a valid scored result")
                row = {
                    "task_id": task["id"],
                    "candidate_sha256": candidate,
                    "score": float(score),
                    "native_metrics": response.get("native_metrics", {}),
                    "usage": response.get("usage", {}),
                    "wall_seconds": time.monotonic() - started,
                }
                json.dumps(row, allow_nan=False)
                self.state["cache"][key] = row
            except Exception as exc:
                self.state["events"].append(
                    {
                        "stage": "task_error",
                        "task_id": task["id"],
                        "candidate_sha256": candidate,
                        "error": str(exc)[:2000],
                    }
                )
                self.save()
                raise
            self.save()

        for offset in range(0, len(missing), 2):
            completed = await asyncio.gather(
                *(execute(task) for task in missing[offset : offset + 2]), return_exceptions=True
            )
            for item in completed:
                if isinstance(item, BaseException):
                    raise item


class SlicedStudy:
    """Run orchestration. Injectable executors support offline lifecycle tests."""

    def __init__(
        self,
        study: Path,
        output: Path,
        seed: int,
        runtime_image: str,
        *,
        task_executor=None,
        search_executor=None,
        adapter: Path = ADAPTER,
        source_identity: dict | None = None,
        required_benchmarks: list[str] | None = None,
    ):
        self.study, self.output, self.seed = study.resolve(), output.resolve(), seed
        if self.output == self.study or self.study.is_relative_to(self.output):
            raise ValueError("Run output must be separate from prepared study inputs")
        self.data = load_study(self.study)
        self.policy = self.data["summary"]["protocol"]["policy"]
        self.runtime_image, self.adapter = runtime_image, adapter.resolve()
        self.task_executor, self.search_executor = task_executor, search_executor
        self.required_benchmarks = required_benchmarks
        self.identity = {
            "study_sha256": self.data["summary"]["study_sha256"],
            "seed": seed,
            "runtime_image": runtime_image,
            "settings": SETTINGS,
            "sources": source_identity
            if source_identity is not None
            else _source_identity(self.adapter),
        }
        self.signature = digest(self.identity)
        self.state: dict[str, Any] = {}
        self.original = self.output / "original_seed.py"
        self.tracker: Any = None

    def save(self):
        self.state["updated_utc"] = utcnow()
        self.state["task_attempts"] = sum(
            read_json(path).get("task_evaluations", 0)
            for path in self.output.rglob("evaluation_state.json")
        )
        if self.state["task_attempts"] > SETTINGS["maximum_task_attempts"]:
            raise RuntimeError("Global task-attempt cap exceeded")
        write_json(self.output / "run.json", self.state)

    def _stage(self, directory: Path, panel: dict, role: str) -> Path:
        if role not in {"search", "selection"} or panel["split"] == "final":
            raise ValueError("Final evaluation is prohibited in this launcher")
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {**panel, "protocol": digest([self.signature, panel["protocol"], role])}
        panel_path = directory / "panel.json"
        config = {
            "policy": self.policy,
            "runtime_image": self.runtime_image,
            "panel_path": str(panel_path),
            "execution_role": role,
            "study_signature": self.signature,
        }
        for path, expected in ((panel_path, manifest), (directory / "run.json", config)):
            if path.exists() and read_json(path) != expected:
                raise ValueError("Stage panel/config identity changed on resume")
            write_json(path, expected)
        return panel_path

    def _task_fn(self):
        if self.task_executor is not None:
            return self.task_executor
        from moevo.generation.evaluator import load_evaluate_fn

        return load_evaluate_fn(str(self.adapter), "evaluate_task")

    async def _full_panel(
        self, directory: Path, panel: dict, role: str, code_path: Path, cap: int
    ) -> dict:
        self._stage(directory, panel, role)
        ledger = PanelLedger(
            directory / "evaluation_state.json",
            digest([self.signature, panel, role]),
            panel["tasks"],
            cap,
            self._task_fn(),
            {key: self.policy[key] for key in ("judge_model", "judge_reasoning_effort")},
        )
        with stage_environment(directory):
            await ledger.ensure(code_path)
        self.save()
        return ledger.state["cache"]

    @contextlib.contextmanager
    def _proposal_ledger(self, search_controller=None):
        import moevo.controller as controller
        from moevo.generation.llm import generate as generic_generate

        original = controller.generate
        original_builder, original_parser = controller.build_prompt, controller.parse_response
        pending: dict[str, Any] = {}

        def instruction_prompt(parent, context, objectives, diff_mode=True, explore=False):
            system, user = build_instruction_prompt(parent, context, objectives, diff_mode, explore)
            pending.clear()
            pending.update(
                parent=parent.to_dict(),
                context=[program.to_dict() for program in context],
                objectives=list(objectives),
                explore=explore,
            )
            return system, user

        def strict_parse(response, parent_code=None, diff_mode=False):
            parent = pending.get("parent", {}).get("solution")
            if parent is None or (parent_code is not None and parent_code != parent):
                raise ValueError("Proposal parent identity changed before parsing")
            candidate = parse_candidate_response(response, parent)
            validate_instruction_change(parent, candidate)
            return candidate

        async def counted(**kwargs):
            if self.state["proposal_calls"] >= SETTINGS["maximum_proposal_calls"]:
                raise EvaluationBudgetError("Fixed proposal-call cap exhausted")
            if kwargs.get("model") != "codex/gpt-6-astra":
                raise ValueError(
                    "Instruction mutation requires account Astra; API fallback forbidden"
                )
            if not pending:
                raise ValueError(
                    "Instruction mutation requires a captured parent and search context"
                )
            if original is generic_generate and os.environ.get("MOEVO_ACCOUNT_ONLY") != "1":
                raise ValueError("Instruction mutation requires account-only stage context")
            self.state["proposal_calls"] += 1
            number = self.state["proposal_calls"]
            directory = self.output / "proposals" / f"call-{number:04d}"
            directory.mkdir(parents=True, exist_ok=False)
            parent = pending["parent"]["solution"]
            schedule = getattr(search_controller, "_schedule", None)
            screen_ids = (
                schedule.screen_ids(schedule.state["screens"]) if schedule is not None else None
            )
            event = {
                "call": number,
                "slice": self.state["active_slice"],
                "status": "started",
                "started_utc": utcnow(),
                "artifact_dir": str(directory.relative_to(self.output)),
                "proposal_budget_charge": 1,
                "parent_source_sha256": source_sha256(parent),
                "usage_status": "not_yet_returned",
            }
            request = {
                **pending,
                "protocol": PROTOCOL,
                "system": kwargs["system"],
                "user": kwargs["user"],
                "model": "gpt-6-astra",
                "reasoning_effort": "xhigh",
                "authentication": "codex_chatgpt_account",
                "transport": "codex_cli"
                if original is generic_generate
                else "injected_test_executor",
                "panel": self.data["panels"][self.state["active_slice"]],
                "next_screen_ids": screen_ids,
                "screen_index": schedule.state["screens"] if schedule is not None else None,
                "parent_source_sha256": source_sha256(parent),
                "proposal_budget_charge": 1,
            }
            # Both provenance and the charge are durable before invoking the model.
            (directory / "parent.py").write_text(parent)
            write_json(directory / "request.json", request)
            write_json(directory / "status.json", event)
            self.state["proposal_events"].append(event)
            self.save()
            try:
                if original is generic_generate:
                    result = await generate_account_proposal(
                        kwargs["system"], kwargs["user"], directory, kwargs.get("timeout", 300)
                    )
                    response = result.text
                    event.update(
                        usage=result.usage,
                        usage_status="reported_by_codex",
                        duration_seconds=result.duration_s,
                    )
                    write_json(directory / "usage.json", result.usage)
                    (directory / "events.jsonl").write_text(
                        "".join(json.dumps(item) + "\n" for item in result.events)
                    )
                else:
                    response = await original(**kwargs)
                    event["usage_status"] = "unavailable_injected_text_executor"
                (directory / "raw_response.txt").write_text(response)
                event["status"] = "returned"
                write_json(directory / "status.json", event)
                self.save()
                candidate = parse_candidate_response(response, parent)
                (directory / "parsed_candidate.py").write_text(candidate)
                (directory / "candidate.diff").write_text(
                    "".join(
                        difflib.unified_diff(
                            parent.splitlines(keepends=True),
                            candidate.splitlines(keepends=True),
                            fromfile="parent.py",
                            tofile="parsed_candidate.py",
                        )
                    )
                )
                validate_instruction_change(parent, candidate)
                event.update(
                    status="validated_before_evaluation",
                    candidate_source_sha256=source_sha256(candidate),
                    completed_utc=utcnow(),
                )
                write_json(directory / "status.json", event)
                self.save()
                return response
            except Exception as exc:
                event.update(
                    status="rejected" if event["status"] == "returned" else "failed",
                    error=str(exc)[:1000],
                    completed_utc=utcnow(),
                )
                if event["usage_status"] == "not_yet_returned":
                    event["usage_status"] = "unavailable_failed_call"
                write_json(directory / "status.json", event)
                self.save()
                raise

        controller.generate, controller.build_prompt, controller.parse_response = (
            counted,
            instruction_prompt,
            strict_parse,
        )
        try:
            yield
        finally:
            controller.generate, controller.build_prompt, controller.parse_response = (
                original,
                original_builder,
                original_parser,
            )

    def _config(self, directory: Path, manifest: Path, initial: Path, index: int) -> MoevoConfig:
        panel = self.data["panels"][SLICE_NAMES[index]]
        return MoevoConfig(
            initial_program=str(initial),
            evaluator_path=str(self.adapter),
            objectives=list(dict.fromkeys(task["objective"] for task in panel["tasks"])),
            model="codex/gpt-6-astra",
            iterations=4,
            population_size=24,
            num_islands=2,
            selection="nsga3",
            retry_attempts=0,
            output_dir=str(directory),
            evaluation_manifest=str(manifest),
            screen_domains=3,
            screen_tasks_per_domain=1,
            screen_audit_every=4,
            max_task_evaluations=5 * len(panel["tasks"]),
            evaluation_concurrency=2,
            judge_model=self.policy["judge_model"],
            judge_reasoning_effort=self.policy["judge_reasoning_effort"],
            random_seed=self.seed + index,
            fresh_start=not (directory / "evaluation_state.json").exists(),
        )

    def _metric_event(self, name: str, entry: dict):
        row = {
            "slice": name,
            "search_seed": self.seed,
            "task_attempts": self.state["task_attempts"],
            "proposal_calls": self.state["proposal_calls"],
            "slice_index": int(name[1:]),
            "pareto_size": entry["pareto_size"],
            "search_hypervolume": entry["hypervolume"],
            "wall_seconds": time.monotonic() - self.started,
            "development_same_panel_delta_pp": entry["development"]["overall"]["delta_pp"],
            "selection_fixed_panel_delta_pp": entry["selection"]["overall"]["delta_pp"],
        }
        with (self.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if self.tracker is not None:
            self.tracker.log(row)

    async def run(self, *, resume: bool = False, through_slice: int = 8) -> dict:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.output.parent / f".{self.output.name}.study.lock"
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another sliced-study runner holds this output lock") from exc
            try:
                return await self._run(resume=resume, through_slice=through_slice)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    async def _run(self, *, resume: bool = False, through_slice: int = 8) -> dict:
        check = readiness(
            self.study, through_slice=through_slice, required_benchmarks=self.required_benchmarks
        )
        if not check["ready"]:
            raise ValueError("Study is not ready: " + "; ".join(check["blockers"]))
        self.started = time.monotonic()
        path = self.output / "run.json"
        if path.exists():
            if not resume:
                raise ValueError("Existing run requires --resume")
            self.state = read_json(path)
            if self.state.get("signature") != self.signature:
                raise ValueError("Run source/config/panel identity changed; refusing stale resume")
        else:
            if resume or (self.output.exists() and any(self.output.iterdir())):
                raise ValueError("New run requires an empty output directory")
            self.output.mkdir(parents=True, exist_ok=True)
            self.original.write_text(
                '"""Original shared instruction seed."""\nINSTRUCTIONS = '
                + repr(self.policy["instructions"])
                + "\n"
            )
            self.state = {
                "signature": self.signature,
                "identity": self.identity,
                "status": "prepared",
                "created_utc": utcnow(),
                "original_seed_sha256": digest(self.original.read_text()),
                "slices": {},
                "proposal_calls": 0,
                "proposal_events": [],
                "resume_semantics": "checkpoint resume; pending proposals are retained as artifacts but not replayed; repeated calls remain charged",
                "final_evaluation_allowed": False,
                "readiness": check,
            }
            self.save()
        if digest(self.original.read_text()) != self.state["original_seed_sha256"]:
            raise ValueError("Original seed source changed")
        self.state.update(status="running", readiness=check)
        self.state.pop("error", None)
        self.save()
        carry = self.original
        try:
            for index, name in enumerate(SLICE_NAMES[:through_slice]):
                panel = self.data["panels"][name]
                entry = self.state["slices"].setdefault(name, {})
                directory = self.output / "slices" / name
                search_dir = directory / "search"
                manifest = self._stage(search_dir, panel, "search")
                initial = search_dir / "initial.py"
                carry_code = carry.read_text()
                if initial.exists() and initial.read_text() != carry_code:
                    raise ValueError("Carry-forward source lineage changed")
                initial.write_text(carry_code)
                entry.setdefault("incoming_sha256", digest(carry_code))
                if entry["incoming_sha256"] != digest(carry_code):
                    raise ValueError("Saved incoming harness differs from previous slice output")
                self.state["active_slice"] = name
                self.save()
                chosen = directory / "chosen.py"
                if not entry.get("search_complete"):
                    config = self._config(search_dir, manifest, initial, index)
                    config_snapshot = {
                        key: value for key, value in asdict(config).items() if key != "fresh_start"
                    }
                    if "search_config" in entry and entry["search_config"] != config_snapshot:
                        raise ValueError("Search configuration changed")
                    entry["search_config"] = config_snapshot
                    self.save()
                    engine = MoevoController(config)
                    with stage_environment(search_dir), self._proposal_ledger(engine):
                        result = await (
                            self.search_executor(config)
                            if self.search_executor is not None
                            else engine.run()
                        )
                    if (
                        result.best_program is None
                        or result.iterations_completed != 4
                        or result.stop_reason != "iterations_completed"
                    ):
                        entry["stop_reason"] = result.stop_reason
                        raise EvaluationBudgetError(
                            "Slice search did not finish all four planned mutation steps"
                        )
                    chosen.write_text(result.best_program.solution)
                    entry.update(
                        search_complete=True,
                        chosen_sha256=digest(result.best_program.solution),
                        iterations_completed=result.iterations_completed,
                        pareto_size=len(result.pareto_front),
                        chosen_program_id=result.best_program.id,
                        hypervolume=result.hypervolume,
                        search_ledger_sha256=digest(
                            read_json(search_dir / "evaluation_state.json")
                        ),
                    )
                    self.save()
                elif not chosen.exists() or digest(chosen.read_text()) != entry["chosen_sha256"]:
                    raise ValueError("Completed slice chosen source changed or disappeared")
                search_ledger = read_json(search_dir / "evaluation_state.json")
                if digest(search_ledger) != entry["search_ledger_sha256"]:
                    raise ValueError("Completed search evaluation ledger changed")
                cache = search_ledger["cache"]
                original_hash, chosen_hash = (
                    digest(self.original.read_text()),
                    digest(chosen.read_text()),
                )
                present = {
                    row["task_id"]
                    for row in cache.values()
                    if row["candidate_sha256"] == original_hash
                }
                if present != {task["id"] for task in panel["tasks"]}:
                    own = await self._full_panel(
                        directory / "original_seed",
                        panel,
                        "search",
                        self.original,
                        len(panel["tasks"]),
                    )
                    # Only take missing observations; no duplicate candidate/task rows.
                    cache = {
                        **cache,
                        **{key: row for key, row in own.items() if row["task_id"] not in present},
                    }
                development = aggregate_panel(panel["tasks"], cache, original_hash, chosen_hash)
                if entry.get("complete") and entry["development"] != development:
                    raise ValueError("Completed same-panel comparison changed")
                entry["development"] = development
                if entry["development"]["overall"]["candidate"] is None:
                    raise ValueError("Chosen harness lacks a complete current-slice evaluation")
                monitor = self.data["panels"]["selection"]
                monitor_dir = self.output / "selection_monitor"
                await self._full_panel(
                    monitor_dir, monitor, "selection", self.original, 9 * len(monitor["tasks"])
                )
                monitored = await self._full_panel(
                    monitor_dir, monitor, "selection", chosen, 9 * len(monitor["tasks"])
                )
                selection = aggregate_panel(monitor["tasks"], monitored, original_hash, chosen_hash)
                if entry.get("complete") and entry["selection"] != selection:
                    raise ValueError("Completed fixed-monitor comparison changed")
                entry["selection"] = selection
                entry["complete"] = True
                entry["completed_utc"] = entry.get("completed_utc", utcnow())
                self.save()
                write_json(
                    directory / "comparison.json",
                    {
                        "development_same_panel": entry["development"],
                        "selection_fixed_panel": entry["selection"],
                    },
                )
                if not entry.get("metrics_logged"):
                    self._metric_event(name, entry)
                    entry["metrics_logged"] = True
                    self.save()
                carry = chosen
            self.state["completed_slices"] = [
                name for name in SLICE_NAMES if self.state["slices"].get(name, {}).get("complete")
            ]
            self.state["status"] = (
                "completed" if len(self.state["completed_slices"]) == 8 else "partial_complete"
            )
            latest = self.state["completed_slices"][-1]
            self.state["latest_development_chosen_sha256"] = self.state["slices"][latest][
                "chosen_sha256"
            ]
            self.save()
            return self.state
        except Exception as exc:
            self.state.update(
                status="stopped_budget" if isinstance(exc, EvaluationBudgetError) else "failed",
                error=str(exc)[:2000],
            )
            self.save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--through-slice", choices=SLICE_NAMES, default="S8")
    parser.add_argument("--image", default="docker.io/library/moevo-or-runtime:20260915")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline"], default="offline")
    args = parser.parse_args()
    through = SLICE_NAMES.index(args.through_slice) + 1
    check = readiness(args.study, through_slice=through)
    if not ADAPTER.exists():
        check["ready"] = False
        check["blockers"].append("Missing exact-task sliced adapter")
    if args.dry_run:
        print(json.dumps(check, indent=2))
        return
    if not check["ready"]:
        parser.error("; ".join(check["blockers"]))
    if args.wandb_mode == "offline":
        import wandb
    from moevo.codex.client import require_chatgpt_login

    require_chatgpt_login()
    image = json.loads(
        subprocess.check_output(["docker", "image", "inspect", args.image], text=True, timeout=30)
    )[0]["Id"]
    runner = SlicedStudy(args.study, args.output, args.seed, image)
    if args.wandb_mode == "offline":
        # Offline segments have unique IDs; durable metrics.jsonl provides the
        # complete cross-resume record. No task prompts or judge feedback logged.
        track_dir = args.output.parent / (args.output.name + "-tracking")
        track_dir.mkdir(parents=True, exist_ok=True)
        runner.tracker = wandb.init(
            project="moevo-sliced-development",
            group=runner.signature[:16],
            name=f"seed-{args.seed}",
            mode="offline",
            dir=str(track_dir),
            config=runner.identity,
            settings=wandb.Settings(disable_git=True, save_code=False, console="off"),
        )
    try:
        result = asyncio.run(runner.run(resume=args.resume, through_slice=through))
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("status", "completed_slices", "task_attempts", "proposal_calls")
                },
                indent=2,
            )
        )
    finally:
        if runner.tracker is not None:
            runner.tracker.finish(
                exit_code=0
                if runner.state.get("status") in {"completed", "partial_complete"}
                else 1
            )


if __name__ == "__main__":
    main()
