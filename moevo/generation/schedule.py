"""Paired search screens followed by one fixed, complete comparison panel.

The task adapter owns execution and grading. This module only allocates calls;
it never turns partial scores or infrastructure failures into population fitness.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import random
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..codex.judging import judge_options
from ..core.types import EvalResult

if TYPE_CHECKING:
    from collections.abc import Callable


class EvaluationBudgetError(RuntimeError):
    """The next complete evaluation stage does not fit the remaining task budget."""


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class StagedEvaluator:
    """Opt-in scheduler for evaluate_task(program_path, task) adapters.

    A task result must contain status='scored' and a finite score in [0, 1].
    Optional feedback is search feedback, never selection/test feedback.
    Cache entries are frozen observations, not independent repeat measurements.
    """

    def __init__(
        self,
        evaluate_task: Callable,
        *,
        manifest_path: str,
        evaluator_path: str,
        objectives: list[str],
        model: str,
        output_dir: Path,
        judge_model: str = "gpt-5.6-terra",
        judge_reasoning_effort: str = "medium",
        concurrency: int = 1,
        screen_domains: int = 3,
        screen_tasks: int = 2,
        audit_every: int = 10,
        max_task_evaluations: int = 1000,
        random_seed: int = 42,
        fresh_start: bool = True,
    ):
        if min(screen_domains, screen_tasks, audit_every, max_task_evaluations) < 1:
            raise ValueError("Screen sizes, audit interval, and task budget must be positive")
        if not objectives or len(set(objectives)) != len(objectives):
            raise ValueError("Objectives must be nonempty and unique")
        if concurrency not in {1, 2}:
            raise ValueError("Task concurrency must be 1 or 2")
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest.get("version") != 1 or manifest.get("split") != "search":
            raise ValueError("Staged evolution requires a version-1 search-only manifest")
        if not isinstance(manifest.get("protocol"), str) or not manifest["protocol"].strip():
            raise ValueError("Manifest must pin its data/model/grader/environment protocol")
        self.tasks: dict[str, dict] = {}
        self.pools: dict[str, list[str]] = {objective: [] for objective in objectives}
        identities = set()
        for task in manifest["tasks"]:
            if any(
                not isinstance(task.get(key), str) or not task[key].strip()
                for key in ("id", "objective", "benchmark", "task_id")
            ):
                raise ValueError("Each task needs nonempty id, objective, benchmark, and task_id")
            if type(task.get("seed")) is not int:
                raise ValueError("Each task needs an explicit integer seed/replicate identifier")
            if task["objective"] not in self.pools:
                raise ValueError(f"Unknown task objective: {task['objective']}")
            identity = (task["benchmark"], task["task_id"], task["seed"])
            if task["id"] in self.tasks or identity in identities:
                raise ValueError("Duplicate task ID or benchmark/task/seed identity")
            # Only these fields cross the adapter boundary; no labels or holdout manifest.
            self.tasks[task["id"]] = {
                key: task[key] for key in ("id", "objective", "benchmark", "task_id", "seed")
            }
            self.pools[task["objective"]].append(task["id"])
            identities.add(identity)
        self.confirmation = manifest["confirmation_ids"]
        if len(set(self.confirmation)) != len(self.confirmation) or any(
            task_id not in self.tasks for task_id in self.confirmation
        ):
            raise ValueError("Confirmation IDs must be unique IDs from the search manifest")
        for objective, pool in self.pools.items():
            if len(pool) < screen_tasks:
                raise ValueError(f"{objective}: not enough distinct tasks for a screen")
            panel = [t for t in self.confirmation if self.tasks[t]["objective"] == objective]
            if not panel:
                raise ValueError(f"Confirmation panel is missing objective {objective}")
            if {self.tasks[t]["benchmark"] for t in panel} != {
                self.tasks[t]["benchmark"] for t in pool
            }:
                raise ValueError(f"Confirmation panel must cover every benchmark in {objective}")
        if fresh_start and max_task_evaluations < len(self.confirmation):
            raise ValueError("Task budget cannot cover the complete seed confirmation panel")

        self.evaluate_task = evaluate_task
        self.concurrency = concurrency
        self.mutation_scope = str(manifest.get("mutation_scope", ""))
        self.judge_policy = {
            "judge_model": judge_model,
            "judge_reasoning_effort": judge_reasoning_effort,
        }
        judge_options(self.judge_policy)
        self.objectives = list(objectives)
        self.screen_domains = min(screen_domains, len(objectives))
        self.screen_tasks = screen_tasks
        self.audit_every = audit_every
        self.max_task_evaluations = max_task_evaluations
        rng = random.Random(random_seed)
        self.domain_order = list(objectives)
        rng.shuffle(self.domain_order)
        for pool in self.pools.values():
            rng.shuffle(pool)
        self.signature = _digest(
            {
                "scheduler_version": 1,
                "manifest": manifest,
                "evaluator_sha256": hashlib.sha256(Path(evaluator_path).read_bytes()).hexdigest(),
                "objectives": objectives,
                "model": model,
                "judge": self.judge_policy,
                "concurrency": concurrency,
                "screen_domains": self.screen_domains,
                "screen_tasks": screen_tasks,
                "audit_every": audit_every,
                "random_seed": random_seed,
            }
        )
        self.state_path = output_dir / "evaluation_state.json"
        self.state: dict = {
            "signature": self.signature,
            "task_evaluations": 0,
            "screens": 0,
            "cache": {},
            "seen": [],
            "events": [],
        }
        if self.state_path.exists():
            if fresh_start:
                raise ValueError("Use a new output directory or --resume for staged evolution")
            self.state = json.loads(self.state_path.read_text())
            if self.state.get("signature") != self.signature:
                raise ValueError(
                    "Evaluation protocol changed; use a new run and re-evaluate scores"
                )
        elif not fresh_start and list(output_dir.glob("checkpoint_*.json")):
            raise ValueError("Cannot resume staged evolution without its evaluation ledger")
        self._save()

    @property
    def task_evaluations(self) -> int:
        return self.state["task_evaluations"]

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False))
        temporary.replace(self.state_path)

    def _key(self, code: str, task_id: str) -> str:
        return _digest([self.signature, code, task_id])

    def seen(self, code: str) -> bool:
        return _digest(code) in self.state["seen"]

    def _require_budget(self, requests: list[tuple[str, str]]) -> None:
        missing = {
            self._key(code, task_id)
            for code, task_id in requests
            if self._key(code, task_id) not in self.state["cache"]
        }
        remaining = self.max_task_evaluations - self.task_evaluations
        if len(missing) > remaining:
            raise EvaluationBudgetError(
                f"Next stage needs {len(missing)} uncached task evaluations; {remaining} remain"
            )

    def screen_ids(self, index: int) -> list[str]:
        """Round-robin domains and tasks, reproducible without global random state."""
        selected = []
        for offset in range(self.screen_domains):
            position = index * self.screen_domains + offset
            objective = self.domain_order[position % len(self.domain_order)]
            visit = position // len(self.domain_order)
            pool = self.pools[objective]
            selected.extend(
                pool[(visit * self.screen_tasks + j) % len(pool)] for j in range(self.screen_tasks)
            )
        return selected

    async def _batch(self, code: str, ids: list[str]) -> dict[str, dict]:
        self._require_budget([(code, task_id) for task_id in ids])
        rows = {}
        for start in range(0, len(ids), self.concurrency):
            completed = await asyncio.gather(
                *(self._batch_sequential(code, [t]) for t in ids[start : start + self.concurrency]),
                return_exceptions=True,
            )
            for item in completed:
                if isinstance(item, BaseException):
                    raise item
                rows.update(item)
        return rows

    async def _batch_sequential(self, code: str, ids: list[str]) -> dict[str, dict]:
        self._require_budget([(code, task_id) for task_id in ids])
        rows = {}
        with tempfile.TemporaryDirectory(prefix="moevo-tasks-") as directory:
            path = Path(directory) / "candidate.py"
            path.write_text(code)
            for task_id in ids:
                key = self._key(code, task_id)
                if key not in self.state["cache"]:
                    # Charge before executing: failures and interrupted calls consume budget too.
                    self.state["task_evaluations"] += 1
                    self._save()
                    started = time.monotonic()
                    try:
                        result = await asyncio.to_thread(
                            self.evaluate_task,
                            str(path),
                            {**self.tasks[task_id], **self.judge_policy},
                        )
                        if inspect.isawaitable(result):
                            result = await result
                        score = result.get("score") if isinstance(result, dict) else None
                        if (
                            not isinstance(result, dict)
                            or result.get("status") != "scored"
                            or isinstance(score, bool)
                            or not isinstance(score, (int, float))
                            or not math.isfinite(score)
                            or not 0 <= score <= 1
                        ):
                            raise ValueError(f"Task {task_id} has no valid scored result: {result}")
                        row = {
                            "task_id": task_id,
                            "candidate_sha256": _digest(code),
                            "score": float(score),
                            "feedback": str(result.get("feedback", ""))[:2000],
                            "wall_seconds": time.monotonic() - started,
                            "native_metrics": result.get("native_metrics", {}),
                            "usage": result.get("usage", {}),
                        }
                        json.dumps(row, allow_nan=False)  # Reject invalid artifacts before caching.
                        self.state["cache"][key] = row
                    except Exception as exc:
                        self.state["events"].append(
                            {"stage": "task_error", "task_id": task_id, "error": str(exc)[:2000]}
                        )
                        self._save()
                        raise
                    self._save()
                rows[task_id] = self.state["cache"][key]
        return rows

    def _metrics(self, rows: dict[str, dict]) -> dict[str, float]:
        return {
            objective: sum(row["score"] for row in selected) / len(selected)
            for objective in self.objectives
            if (
                selected := [
                    row
                    for task_id, row in rows.items()
                    if self.tasks[task_id]["objective"] == objective
                ]
            )
        }

    async def confirm(self, code: str) -> EvalResult:
        """Same full search panel for the seed and every population candidate."""
        rows = await self._batch(code, self.confirmation)
        feedback = "\n".join(
            f"{task_id}: {row['feedback']}" for task_id, row in rows.items() if row["feedback"]
        )
        return EvalResult(
            metrics=self._metrics(rows),
            artifacts={"feedback": feedback, "evaluation_signature": self.signature},
        )

    async def candidate(
        self, code: str, parent_code: str, parent_metrics: dict[str, float]
    ) -> EvalResult | None:
        """Return a complete vector, or None for a screened-out candidate."""
        ids = self.screen_ids(self.state["screens"])
        self._require_budget([(c, t) for c in (parent_code, code) for t in ids])
        self.state["screens"] += 1
        self._save()
        parent = self._metrics(await self._batch(parent_code, ids))
        child = self._metrics(await self._batch(code, ids))
        delta = {objective: child[objective] - parent[objective] for objective in child}
        improved = any(value > 1e-12 for value in delta.values())
        audit = self.state["screens"] % self.audit_every == 0
        event = {
            "stage": "screen",
            "candidate_sha256": _digest(code),
            "screen": self.state["screens"],
            "task_ids": ids,
            "delta": delta,
            "improved": improved,
            "audit": audit,
            "confirmed": False,
        }
        self.state["events"].append(event)
        self._save()
        result = None
        if improved or audit:
            result = await self.confirm(code)
            event["confirmed"] = True
            event["confirmation_delta"] = {
                objective: result.metrics[objective] - parent_metrics[objective]
                for objective in self.objectives
            }
            event["screen_missed_improvement"] = not improved and any(
                value > 1e-12 for value in event["confirmation_delta"].values()
            )
        self.state["seen"].append(_digest(code))
        self._save()
        return result
