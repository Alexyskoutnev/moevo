"""Read-only task progress from the frozen ledger and execution artifacts.

An attempt directory proves dispatch, not completion or liveness. Only a cached
score counts as scored, and only an independently observed controller supports
an active-process label. This observer never constructs or advances a scheduler.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .source import code_hash


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def controller_liveness(root: Path) -> dict:
    """Inspect only command identities; never read credentials or process envs."""
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {"state": "unknown", "pids": []}
    repository = Path(__file__).resolve().parents[2]
    pids = []
    for line in result.stdout.splitlines():
        try:
            pid, command = line.strip().split(maxsplit=1)
            parts = shlex.split(command)
            module = parts[parts.index("-m") + 1]
            if module not in {
                "experiments.run_first_slice",
                "experiments.run_pilot_epoch",
                "experiments.run_reference_baselines",
                "experiments.run_reference_parallel",
            }:
                continue
            if "--_worker" in parts:
                continue
            output = Path(parts[parts.index("--output") + 1])
            output = output if output.is_absolute() else repository / output
            if output.resolve() == root.resolve():
                pids.append(int(pid))
        except (ValueError, IndexError):
            continue
    return {"state": "active" if pids else "stopped", "pids": sorted(pids)}


def screen_task_ids(manifest: dict, config: dict, index: int) -> list[str]:
    """Reconstruct scheduler v1's deterministic rotation without side effects."""
    objectives = config.get("objectives", [])
    if not objectives or index < 0:
        return []
    pools = {
        o: [t["id"] for t in manifest.get("tasks", []) if t["objective"] == o] for o in objectives
    }
    if any(not p for p in pools.values()):
        return []
    rng = random.Random(config.get("random_seed", 42))
    order = list(objectives)
    rng.shuffle(order)
    for pool in pools.values():
        rng.shuffle(pool)
    domains = min(config.get("screen_domains", 3), len(order))
    count = config.get("screen_tasks_per_domain", 2)
    ids = []
    for offset in range(domains):
        position = index * domains + offset
        pool = pools[order[position % len(order)]]
        visit = position // len(order)
        ids.extend(pool[(visit * count + j) % len(pool)] for j in range(count))
    return ids


def _iso(timestamp: float | None) -> str | None:
    return datetime.fromtimestamp(timestamp, UTC).isoformat() if timestamp is not None else None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime if not path.is_symlink() else None
    except OSError:
        return None


def _attempts(root: Path, tasks: list[dict], state: dict, sources: dict) -> list[dict]:
    raw = {}
    for sha, code in sources.items():
        prefix = hashlib.sha256(code.encode()).hexdigest()[:12]
        raw.setdefault(prefix, []).append(sha)
    by_artifact = {
        str((root / r.get("usage", {}).get("artifact_path", "")).resolve()): r
        for r in state.get("cache", {}).values()
    }
    attempts = []
    for path in sorted((root / "tasks").glob("*/*")):
        if not path.is_dir() or not path.resolve().is_relative_to(root.resolve()):
            continue
        prefix = path.name.split("-")[0]
        if not re.fullmatch("[0-9a-f]{12}", prefix):
            continue
        cached = by_artifact.get(str(path.resolve()))
        matching = [t for t in tasks if t["benchmark"] == path.parent.name]
        identity = _json(path / "preflight.json")
        published = identity.get("task_id", identity.get("id"))
        if published:
            matching = [t for t in matching if t["task_id"] == published]
        task_id = cached["task_id"] if cached else matching[0]["id"] if len(matching) == 1 else None
        # Do not scan copied benchmark assets or count browser/observer refreshes.
        files = [f for f in path.glob("*") if f.is_file()]
        for folder in (
            "agent",
            "grade",
            "calls",
            "controls",
            "positive_control",
            "negative_control",
        ):
            files.extend((path / folder).rglob("*.json"))
            files.extend((path / folder).rglob("*.jsonl"))
        times = [t for f in files if (t := _mtime(f)) is not None]
        modified = max(times, default=_mtime(path))
        candidates = raw.get(prefix, [])
        sha = (
            cached["candidate_sha256"]
            if cached
            else candidates[0]
            if len(candidates) == 1
            else None
        )
        phase = (
            "grading"
            if (path / "extracted.json").exists() or (path / "grade").exists()
            else "solving"
        )
        attempts.append(
            {
                "task_id": task_id,
                "candidate_sha256": sha,
                "raw_prefix": prefix,
                "artifact_path": str(path.relative_to(root)),
                "scored": cached is not None,
                "error": (path / "error.json").exists(),
                "report_written": (path / "report.json").exists(),
                "phase": phase,
                "last_activity": modified,
            }
        )
    return attempts


def task_progress(
    root: Path,
    run: dict,
    manifest: dict,
    state: dict,
    database: dict,
    *,
    process: dict | None = None,
) -> dict:
    tasks, config = manifest.get("tasks", []), run.get("config", {})
    cache = list(state.get("cache", {}).values())
    process = process or {"state": "unknown", "pids": []}
    sources = {code_hash(p["solution"]): p["solution"] for p in database.get("all_programs", [])}
    seed = (root / "seed.py").read_text() if (root / "seed.py").exists() else None
    seed_sha = code_hash(seed) if seed is not None else None
    if seed is not None:
        sources[code_hash(seed)] = seed
    known = {r["candidate_sha256"] for r in cache}
    for path in (root / "source_observations").glob("*.py"):
        code = path.read_text()
        sha = code_hash(code)
        if sha == path.stem and sha in known:
            sources[sha] = code
    attempts = _attempts(root, tasks, state, sources)
    events = [e for e in state.get("events", []) if e.get("stage") == "screen"]
    latest = events[-1] if events else {}
    number = state.get("screens", 0)
    confirmation = manifest.get("confirmation_ids", [t["id"] for t in tasks])
    seed_rows = {r["task_id"]: r for r in cache if r["candidate_sha256"] == seed_sha}
    pending_screen = number > latest.get("screen", 0)
    phase, panel, candidate = "baseline", confirmation, seed_sha
    raw_prefix = None
    if number:
        panel = screen_task_ids(manifest, config, number - 1)
        if pending_screen:
            phase = "quick_check"
            finished = {e.get("candidate_sha256") for e in events}
            finished.update(code_hash(p["solution"]) for p in database.get("all_programs", []))
            unseen = (
                {r["candidate_sha256"] for r in cache if r["task_id"] in panel}
                - finished
                - {seed_sha}
            )
            candidate = next(iter(unseen)) if len(unseen) == 1 else None
            # Before the first score, only the adapter's source prefix is known.
            prefixes = {
                a["raw_prefix"]
                for a in attempts
                if not a["scored"]
                and not a["error"]
                and a["task_id"] in panel
                and a["candidate_sha256"] not in finished | {seed_sha}
            }
            if len(prefixes) == 1:
                raw_prefix = next(iter(prefixes))
                active_hashes = {
                    a["candidate_sha256"]
                    for a in attempts
                    if a["raw_prefix"] == raw_prefix and a["candidate_sha256"]
                }
                candidate = next(iter(active_hashes)) if len(active_hashes) == 1 else None
        else:
            candidate = latest.get("candidate_sha256")
            panel = latest.get("task_ids", panel)
            if latest.get("confirmed") or latest.get("improved") or latest.get("audit"):
                phase, panel = "full_test", confirmation
            else:
                phase = "preparing_change"
    elif confirmation and all(t in seed_rows for t in confirmation):
        phase = "preparing_change"
    candidate_rows = {
        r["task_id"]: r for r in cache if candidate and r["candidate_sha256"] == candidate
    }
    if candidate in sources:
        raw_prefix = hashlib.sha256(sources[candidate].encode()).hexdigest()[:12]
    rows = []
    run_active = run.get("status") == "running"
    for index, task in enumerate(tasks, 1):
        task_id = task["id"]
        row = candidate_rows.get(task_id)
        matches = [
            a
            for a in attempts
            if a["task_id"] == task_id
            and (
                (candidate and a["candidate_sha256"] == candidate)
                or (raw_prefix and a["raw_prefix"] == raw_prefix)
            )
        ]
        attempt = max(matches, key=lambda a: a["last_activity"] or 0, default=None)
        status, detail = "queued", "Waiting for this stage"
        if row:
            status, detail = "scored", "Scored, including zero-credit answers"
        elif attempt and attempt["error"]:
            status, detail = "error", "Infrastructure failure; no score assigned"
        elif attempt:
            if not run_active or process["state"] == "stopped":
                status, detail = "unfinished", "Started; controller is no longer running"
            elif process["state"] == "active":
                status = "running"
                detail = (
                    "Saving result"
                    if attempt["report_written"]
                    else (
                        "Grading the answer"
                        if attempt["phase"] == "grading"
                        else "Solving / task checks"
                    )
                )
            else:
                status, detail = "started", "Started; process status unavailable"
        elif task_id not in panel:
            status, detail = "not_in_check", "Not in this quick check"
        elif not run_active:
            status, detail = "not_run", "Not run in this stage"
        rows.append(
            {
                "index": index,
                "id": task_id,
                "benchmark": task["benchmark"],
                "domain": task["objective"],
                "task_id": task["task_id"],
                "in_stage": task_id in panel,
                "status": status,
                "detail": detail,
                "seed_score": seed_rows.get(task_id, {}).get("score"),
                "score": row.get("score") if row else None,
                "ever_scored": any(r["task_id"] == task_id for r in cache),
                "attempts": sum(a["task_id"] == task_id for a in attempts),
                "artifact_path": attempt["artifact_path"] if attempt else None,
                "last_activity_utc": _iso(attempt["last_activity"]) if attempt else None,
            }
        )
    stage_rows = [r for r in rows if r["in_stage"]]
    timestamps = [a["last_activity"] for a in attempts if a["last_activity"] is not None]
    timestamps.extend(
        t
        for p in [
            root / "evaluation_state.json",
            root / "run.json",
            *root.glob("checkpoint_*.json"),
        ]
        if (t := _mtime(p)) is not None
    )
    return {
        "run_name": root.name,
        "slice_index": 1,
        "slice_total": 1,
        "phase": phase,
        "screen": number,
        "candidate_sha256": candidate,
        "stage_task_ids": panel,
        "stage_total": len(panel),
        "stage_scored": sum(r["status"] == "scored" for r in stage_rows),
        "stage_running": sum(r["status"] == "running" for r in stage_rows),
        "stage_queued": sum(r["status"] == "queued" for r in stage_rows),
        "stage_errors": sum(r["status"] == "error" for r in stage_rows),
        "unique_tasks_scored": sum(r["ever_scored"] for r in rows),
        "total_tasks": len(tasks),
        "scored_evaluations": len(cache),
        "attempts_charged": state.get("task_evaluations", 0),
        "process": process,
        "last_activity_utc": _iso(max(timestamps)) if timestamps else None,
        "tasks": rows,
    }
