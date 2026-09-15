"""Run a real first search slice through the new exact-task adapters.

This bridge run validates new tasks before the complete eight-slice study. Its
Terminal-Bench task is explicitly the existing validated fixture; eight distinct
terminal environments have not yet been validated.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from experiments.prepare_multidomain_study import BENCHMARKS, MUTATION_SCOPE, digest, write_json
from experiments.sliced_task_adapter import backend
from moevo.codex.client import require_chatgpt_login
from moevo.controller import MoevoController
from moevo.core.config import MoevoConfig

ROOT = Path(__file__).resolve().parents[1]
PRECHECKED_IDS = {
    "finqa": "ADI/2009/page_49.pdf-1",
    "bizfinbench2": "numeric-0000",
    "genebench_pro": "carrier_cnv_pseudogene_residual_risk",
    "amo": "p-subset-0000",
    "travelplanner": "train-0",
    "terminal_bench_2": "regex-log",
}


def prepare(output: Path, seed: int) -> dict:
    from moevo.codex import multitask_core, multitask_extended

    records = [*multitask_core.inventory(), *multitask_extended.inventory()]
    old_tasks = json.loads((ROOT / "results/pilot-epoch/astra-terra-01/search.json").read_text())[
        "tasks"
    ]
    old_ids = {(t["benchmark"], t["task_id"]) for t in old_tasks}
    old_ids.add(("tau3_bench", "retail/1"))
    policy = json.loads(
        (ROOT / "submissions/superharness-astra-terra-pilot/submission.json").read_text()
    )
    tasks = []
    for benchmark in BENCHMARKS:
        eligible = [
            r
            for r in records
            if r["benchmark"] == benchmark
            and r.get("adapter_ready") is True
            and ((benchmark, r["id"]) not in old_ids or benchmark == "terminal_bench_2")
        ]
        if benchmark in PRECHECKED_IDS:
            eligible = [r for r in eligible if r["id"] == PRECHECKED_IDS[benchmark]]
        if not eligible:
            raise ValueError(f"No supported exact task for {benchmark}")
        record = min(eligible, key=lambda r: digest([seed, r["benchmark"], r["id"]]))
        checked = backend(benchmark).preflight(benchmark, record["id"])
        if not checked.get("structural_preflight_passed") or checked.get("adapter_ready") is False:
            raise ValueError(f"Structural preflight failed for {benchmark}/{record['id']}")
        task = {
            "id": benchmark,
            "benchmark": benchmark,
            "task_id": record["id"],
            "objective": record["domain"].replace(" ", "_"),
            "seed": policy["seed"],
            "content_sha256": checked["content_sha256"],
            "input_assets_sha256": checked.get("input_assets_sha256"),
        }
        tasks.append(task)
        write_json(output / "preflight" / f"{benchmark}.json", checked)
    source_paths = [
        *sorted((ROOT / "moevo/codex").glob("*.py")),
        ROOT / "experiments/sliced_task_adapter.py",
        ROOT / "experiments/run_first_slice.py",
    ]
    protocol = {
        "policy": policy,
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "tasks": tasks,
        "scope": "First-slice development validation,12new tasks plus1existing TerminalBench fixture",
    }
    write_json(output / "protocol.json", protocol)
    write_json(
        output / "search.json",
        {
            "version": 1,
            "split": "search",
            "protocol": digest(protocol),
            "mutation_scope": MUTATION_SCOPE,
            "tasks": tasks,
            "confirmation_ids": [t["id"] for t in tasks],
        },
    )
    (output / "seed.py").write_text("INSTRUCTIONS = " + repr(policy["instructions"]) + "\n")
    config = MoevoConfig(
        initial_program=str(output / "seed.py"),
        evaluator_path=str(ROOT / "experiments/sliced_task_adapter.py"),
        objectives=list(dict.fromkeys(t["objective"] for t in tasks)),
        model="codex/gpt-6-astra",
        iterations=4,
        population_size=24,
        num_islands=2,
        selection="nsga3",
        retry_attempts=1,
        output_dir=str(output),
        evaluation_manifest=str(output / "search.json"),
        screen_domains=3,
        screen_tasks_per_domain=1,
        screen_audit_every=4,
        max_task_evaluations=65,
        evaluation_concurrency=2,
        random_seed=seed,
    )
    run = {
        "status": "prepared",
        "purpose": protocol["scope"],
        "policy": policy,
        "config": asdict(config),
        "panel_path": str(output / "search.json"),
        "execution_role": "search",
        "created_utc": datetime.now(UTC).isoformat(),
        "excluded": {},
        "runtime_image": None,
    }
    write_json(output / "run.json", run)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.resume:
        run = json.loads((output / "run.json").read_text())
        protocol = json.loads((output / "protocol.json").read_text())
        changed = [
            p
            for p, sha in protocol["sources"].items()
            if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != sha
        ]
        if changed:
            parser.error(f"Frozen source changed; use a new run: {changed}")
    else:
        if output.exists() and any(output.iterdir()):
            parser.error("Use a new output directory or --resume")
        output.mkdir(parents=True, exist_ok=True)
        run = prepare(output, args.seed)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "output": str(output), "tasks": 13}))
        return
    version = require_chatgpt_login()
    image = json.loads(
        subprocess.check_output(
            ["docker", "image", "inspect", "docker.io/library/moevo-or-runtime:20260915"],
            text=True,
            timeout=30,
        )
    )[0]["Id"]
    if run.get("runtime_image") not in {None, image}:
        parser.error("Runtime image changed")
    config = MoevoConfig(**run["config"])
    config.fresh_start = not (output / "evaluation_state.json").exists()
    run.update(status="running", runtime_image=image, cli_version=version)
    write_json(output / "run.json", run)
    os.environ["MOEVO_ACCOUNT_ONLY"] = "1"
    os.environ["MOEVO_SLICED_DIR"] = str(output)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    try:
        result = asyncio.run(MoevoController(config).run())
        run.update(
            status="completed",
            iterations_completed=result.iterations_completed,
            task_evaluations=result.task_evaluations,
            stop_reason=result.stop_reason,
        )
    except Exception as exc:
        run.update(status="failed", error=str(exc))
        raise
    finally:
        run["updated_utc"] = datetime.now(UTC).isoformat()
        write_json(output / "run.json", run)


if __name__ == "__main__":
    main()
