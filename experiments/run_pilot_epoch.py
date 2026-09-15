"""One diagnostic epoch: rotate through the available validated domain fixtures."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from experiments.run_submission import load_submission
from moevo.codex.client import require_chatgpt_login
from moevo.codex.finance_pilot import write_json
from moevo.controller import MoevoController
from moevo.core.config import MoevoConfig

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    "automationbench": "Inconsistent fixture IDs remain under audit",
    "eduagentbench": "Public release has no runtime/evaluator",
    "terminal_bench_science": "Image access must be repaired before evolution",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    policy_path = ROOT / "submissions/superharness-astra-terra-pilot/submission.json"
    policy = load_submission(policy_path)
    require_chatgpt_login()
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error("Choose a new output directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        report = json.loads(
            (ROOT / "submissions/superharness-astra-v1/validation.json").read_text()
        )
        tasks = []
        for row in report["results"]:
            if row["benchmark"] in EXCLUDED or row["status"] != "scored":
                continue
            objective = row["domain"].lower().replace(" / ", "_").replace(" ", "_")
            tasks.append(
                {
                    "id": row["benchmark"],
                    "objective": objective,
                    "benchmark": row["benchmark"],
                    "task_id": row["task_id"],
                    "seed": policy["seed"],
                }
            )
        objectives = list(dict.fromkeys(t["objective"] for t in tasks))
        iterations = math.ceil(len(objectives) / 3)
        sources = [
            *sorted((ROOT / "moevo/codex").glob("*.py")),
            *sorted((ROOT / "experiments").glob("*.py")),
        ]
        images = json.loads(
            subprocess.check_output(
                ["docker", "image", "inspect", "docker.io/library/moevo-or-runtime:20260915"],
                text=True,
            )
        )
        protocol = {
            "policy": policy,
            "sources": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
            "runtime_image": images[0]["Id"],
            "metric_mapping": "native [0,1] clipped; Harvey criterion pass rate; equal task mean per domain",
            "scope": "single-fixture diagnostic, no generalization/headroom claim",
        }
        write_json(output / "protocol.json", protocol)
        write_json(
            output / "search.json",
            {
                "version": 1,
                "split": "search",
                "protocol": hashlib.sha256(
                    json.dumps(protocol, sort_keys=True).encode()
                ).hexdigest(),
                "mutation_scope": "This pilot evolves only the shared instruction component. Return a Python file containing a single literal INSTRUCTIONS string assignment, optionally a module docstring. Do not add imports, functions, other assignments, or executable statements. Improve general planning, verification and recovery instructions that transfer across domains.",
                "tasks": tasks,
                "confirmation_ids": [t["id"] for t in tasks],
            },
        )
        (output / "seed.py").write_text(
            '"""Shared instruction component; models, tools and budgets stay fixed."""\n\n'
            + "INSTRUCTIONS = "
            + repr(policy["instructions"])
            + "\n"
        )
        config = MoevoConfig(
            initial_program=str(output / "seed.py"),
            evaluator_path=str(ROOT / "experiments/pilot_task_adapter.py"),
            objectives=objectives,
            model="codex/gpt-6-astra",
            iterations=iterations,
            population_size=24,
            num_islands=2,
            selection="nsga3",
            retry_attempts=1,
            output_dir=str(output),
            evaluation_manifest=str(output / "search.json"),
            screen_domains=3,
            screen_tasks_per_domain=1,
            screen_audit_every=iterations,
            max_task_evaluations=len(tasks) * (iterations + 1),
            evaluation_concurrency=2,
            random_seed=policy["seed"],
        )
        run = {
            "purpose": "One diagnostic epoch on existing fixtures; not held-out benchmark improvement",
            "status": "prepared",
            "policy": policy,
            "runtime_image": images[0]["Id"],
            "excluded": EXCLUDED,
            "config": asdict(config),
            "created_utc": datetime.now(UTC).isoformat(),
        }
        write_json(output / "run.json", run)
    run = json.loads((output / "run.json").read_text())
    config = MoevoConfig(**run["config"])
    config.fresh_start = not args.resume
    os.environ["MOEVO_ACCOUNT_ONLY"] = "1"
    os.environ["MOEVO_PILOT_DIR"] = str(output)
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run["status"] = "running"
    write_json(output / "run.json", run)
    try:
        result = asyncio.run(MoevoController(config).run())
        run.update(
            status="completed",
            iterations_completed=result.iterations_completed,
            task_evaluations=result.task_evaluations,
            stop_reason=result.stop_reason,
            best_program=result.best_program.id if result.best_program else None,
        )
    except Exception as exc:
        run.update(status="failed", error=str(exc))
        raise
    finally:
        run["updated_utc"] = datetime.now(UTC).isoformat()
        write_json(output / "run.json", run)


if __name__ == "__main__":
    main()
