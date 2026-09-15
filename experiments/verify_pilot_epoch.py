"""Verify a fresh real epoch's artifacts without interpreting smoke scores as accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.evolution_dashboard import read_json, snapshot
from experiments.pilot_task_adapter import instructions_from_code


def finite_score(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid numeric score")
    return float(value)


def verify(run_root: Path) -> dict:
    data = snapshot(run_root)
    run = read_json(run_root / "run.json", {})
    manifest = read_json(run_root / "search.json", {})
    state = read_json(run_root / "evaluation_state.json", {})
    expected = set(manifest.get("confirmation_ids", []))
    task_map = {t["id"]: t for t in manifest.get("tasks", [])}
    seed = run_root / "seed.py"
    seed_hash = (
        hashlib.sha256(json.dumps(seed.read_text(), sort_keys=True).encode()).hexdigest()
        if seed.exists()
        else None
    )
    rows_by_candidate = {}
    missing_artifacts = []
    score_mismatches = []
    for row in state.get("cache", {}).values():
        rows_by_candidate.setdefault(row["candidate_sha256"], {})[row["task_id"]] = row
        artifact = row.get("usage", {}).get("artifact_path")
        report = read_json(run_root / artifact / "report.json", {}) if artifact else {}
        task = task_map.get(row["task_id"], {})
        if not report.get("e2e_validated") or report.get("task_id") != task.get("task_id"):
            missing_artifacts.append(row["task_id"])
        else:
            native = row.get("native_metrics", {}).get("score")
            reported = report.get("score")
            mapped = (
                report.get("criterion_pass_rate", reported)
                if task["benchmark"] == "harvey_lab"
                else reported
            )
            try:
                native, reported, mapped = [finite_score(v) for v in (native, reported, mapped)]
                valid = math.isclose(native, reported, abs_tol=1e-12) and math.isclose(
                    row["score"], max(0, min(1, mapped)), abs_tol=1e-12
                )
            except ValueError:
                valid = False
            if not valid:
                score_mismatches.append(row["task_id"])
    complete = {
        code_hash: rows
        for code_hash, rows in rows_by_candidate.items()
        if expected and expected <= set(rows)
    }
    vector_errors = []
    for candidate in data["history"]:
        rows = complete.get(candidate["candidate_sha256"])
        if rows is None:
            vector_errors.append(candidate["id"])
            continue
        for objective in data["objectives"]:
            scores = [rows[t]["score"] for t in expected if task_map[t]["objective"] == objective]
            mean = sum(scores) / len(scores) if scores else math.nan
            if not math.isclose(mean, candidate["metrics"][objective], abs_tol=1e-12):
                vector_errors.append(candidate["id"])
    best_path = run_root / "best_program.py"
    best_hash = (
        hashlib.sha256(json.dumps(best_path.read_text(), sort_keys=True).encode()).hexdigest()
        if best_path.exists()
        else None
    )
    checkpoints = sorted(run_root.glob("checkpoint_*.json"))
    programs = (
        read_json(checkpoints[-1], {}).get("database", {}).get("all_programs", [])
        if checkpoints
        else []
    )
    changed = set()
    try:
        initial_instructions = instructions_from_code(seed.read_text())
        for program in programs:
            code = program["solution"]
            code_hash = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
            if code_hash in complete and instructions_from_code(code) != initial_instructions:
                changed.add(code_hash)
    except (ValueError, SyntaxError, FileNotFoundError):
        pass
    policy, config = run.get("policy", {}), run.get("config", {})
    checks = {
        "run_completed": data["status"] == "completed",
        "development_tasks_only": manifest.get("split") == "search",
        "complete_seed": seed_hash in complete,
        "configured_astra_generation_and_solving": config.get("model") == "codex/gpt-6-astra"
        and policy.get("model") == "gpt-6-astra"
        and policy.get("reasoning_effort") == "xhigh",
        "configured_terra_rubric_judges": policy.get("judge_model") == "gpt-5.6-terra"
        and policy.get("judge_reasoning_effort") == "medium",
        "configured_account_authentication": policy.get("authentication")
        == "codex_chatgpt_account",
        "real_task_artifacts_match_ids": bool(rows_by_candidate) and not missing_artifacts,
        "ledger_scores_match_raw_grader_reports": bool(rows_by_candidate) and not score_mismatches,
        "at_least_one_mutation_screened": bool(data["screens"]),
        "at_least_one_child_fully_evaluated": bool(changed),
        "population_vectors_reconstruct_from_task_scores": bool(data["history"])
        and not vector_errors,
        "selected_harness_has_complete_scores": best_hash in complete,
        "no_unresolved_task_errors": data["task_errors"] == 0,
        "within_task_budget": data["task_budget"] is not None
        and data["task_evaluations"] <= data["task_budget"],
    }
    return {
        "status": "passed" if all(checks.values()) else "incomplete_or_failed",
        "checks": checks,
        "task_attempts": data["task_evaluations"],
        "task_errors": data["task_errors"],
        "fully_scored_candidates": len(complete),
        "mutations_screened": len(data["screens"]),
        "baseline_mean_percent": 100 * data["baseline"]["mean"] if data["baseline"] else None,
        "selected_mean_percent": 100 * data["champion"]["mean"] if data["champion"] else None,
        "missing_or_mismatched_artifacts": missing_artifacts,
        "score_mismatches": score_mismatches,
        "vector_errors": vector_errors,
        "interpretation": "End-to-end shared-instruction evolution check on existing diagnostic tasks. Passing does not require a score gain and does not establish held-out improvement.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = verify(args.run)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    if args.require_complete and report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
