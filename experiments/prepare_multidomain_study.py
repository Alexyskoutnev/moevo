"""Prepare eight real, group-disjoint development slices without running inference."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from moevo.data.splitters.multidomain import build_multidomain_slices

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = [
    "finqa",
    "bizfinbench2",
    "genebench_pro",
    "amo",
    "putnambench",
    "travelplanner",
    "oragentbench",
    "dsbench",
    "gdpval",
    "terminal_bench_2",
    "harvey_lab",
    "healthbench_professional",
    "tau3_bench",
]
MUTATION_SCOPE = (
    "Evolve only the shared instruction component. Return a Python file with exactly one "
    "literal INSTRUCTIONS string assignment, optionally a module docstring. Do not add "
    "imports, functions, other assignments, executable code, task IDs or reference answers. "
    "Improve general planning, verification and recovery across domains."
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(records: list[dict], identities: dict, *, seed: int, policy: dict) -> dict:
    """One reproducible representative from each of ten families per benchmark.

    Siblings remain assigned to their family's partition but are not all executed
    in this inexpensive first sliced experiment. Selection is a development
    monitor. Final is a reservation only; unknown historical exposure is not fresh.
    """
    # Restrict implemented protocol variants by schema/tool support before any scores exist.
    # Keep Terminal-Bench's unavailable environments visible in the planned panels.
    eligible = [
        r for r in records if r.get("adapter_ready") is True or r["benchmark"] == "terminal_bench_2"
    ]
    eligibility_exclusions = [
        {
            "benchmark": r["benchmark"],
            "id": r["id"],
            "reason": r.get("exclusion_reason", "unsupported adapter"),
        }
        for r in records
        if r not in eligible
    ]
    allocation = build_multidomain_slices(
        eligible,
        seed=seed,
        benchmark_identities=identities,
        expected_benchmarks=BENCHMARKS,
        development_slices=8,
        max_groups_per_benchmark=10,
        final_policy="reserve_only",
    )
    lookup = {(r["benchmark"], r["id"]): r for r in records}
    panels = {name: [] for name in [*[f"S{i}" for i in range(1, 9)], "selection", "final"]}
    for component in allocation["component_assignments"]:
        ref = min(component["records"], key=lambda r: digest([seed, "representative", r]))
        record = lookup[(ref["benchmark"], ref["id"])]
        panels[component["partition"]].append(
            {
                "id": ref["benchmark"] + ":" + digest(ref["id"])[:20],
                "benchmark": ref["benchmark"],
                "task_id": ref["id"],
                "objective": record["domain"],
                "seed": seed,
                "component": component["component"],
                "content_sha256": record["content_sha256"],
                "adapter_ready": record.get("adapter_ready") is True,
            }
        )
    for panel in panels.values():
        panel.sort(key=lambda t: (BENCHMARKS.index(t["benchmark"]), t["task_id"]))
    missing = {
        name: sorted(set(BENCHMARKS) - {t["benchmark"] for t in panel})
        for name, panel in panels.items()
        if set(BENCHMARKS) - {t["benchmark"] for t in panel}
    }
    unsupported = [
        {"partition": name, "benchmark": t["benchmark"], "task_id": t["task_id"]}
        for name, panel in panels.items()
        if name != "final"
        for t in panel
        if not t["adapter_ready"]
    ]
    protocol = {
        "allocation_sha256": allocation["manifest_sha256"],
        "policy": policy,
        "representatives": "One source task per component; deterministic hash, no score filtering",
        "metric_mapping": "native score clipped to [0,1]; Harvey criterion pass rate",
        "weighting": "equal domains, task mean within each domain",
    }
    manifests = {
        name: {
            "version": 1,
            "split": "search" if name.startswith("S") else name,
            "slice": name,
            "protocol": digest(protocol),
            "mutation_scope": MUTATION_SCOPE,
            "tasks": tasks,
            "confirmation_ids": [t["id"] for t in tasks],
        }
        for name, tasks in panels.items()
    }
    counts = {name: len(panel) for name, panel in panels.items()}
    # Upper bounds for four mutations, every child confirmed, on each new slice.
    search_cap = sum(5 * counts[f"S{i}"] for i in range(1, 9))
    baseline_cap = sum(counts[f"S{i}"] for i in range(2, 9))
    monitor_cap = 9 * counts["selection"]
    summary = {
        "version": 1,
        "name": "SuperHarness eight-slice development study",
        "status": "prepared" if not missing and not unsupported else "preparation_incomplete",
        "split_seed": seed,
        "development_slices": 8,
        "mutations_per_slice": 4,
        "mutation_steps": 32,
        "panel_task_counts": counts,
        "benchmarks": BENCHMARKS,
        "domains": sorted({r["domain"] for r in records}),
        "available_tasks": dict(Counter(r["benchmark"] for r in records)),
        "eligible_tasks": dict(Counter(r["benchmark"] for r in eligible)),
        "eligibility_exclusions": eligibility_exclusions,
        "missing_benchmarks": missing,
        "unsupported_tasks": unsupported,
        "structural_readiness_only": True,
        "new_tasks_e2e_validated": False,
        "final_evaluation_allowed": False,
        "final_status": "Reserved only; historical exposure is unknown. Freshness and selection freeze required.",
        "selection_status": "Fixed development monitor; never a final-test score",
        "task_attempt_caps": {
            "evolution": search_cap,
            "additional_unevolved_seed": baseline_cap,
            "selection_monitor": monitor_cap,
            "total": search_cap + baseline_cap + monitor_cap,
        },
        "resource_note": "Task attempts are not model calls. Judges, simulators, controls and proposal calls add work.",
        "protocol": protocol,
    }
    summary["study_sha256"] = digest({"summary": summary, "manifests": manifests})
    return {"summary": summary, "allocation": allocation, "manifests": manifests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory; never overwrite an allocated study")
    from moevo.codex import multitask_core, multitask_extended

    records = [
        {**r, "domain": r["domain"].replace(" ", "_")}
        for r in [*multitask_core.inventory(), *multitask_extended.inventory()]
    ]
    # Current-session task manifests identify known development use without inspecting old results.
    exposed_ids = set()
    for manifest in [
        ROOT / "results/pilot-epoch/astra-terra-01/search.json",
        *sorted((ROOT / "results/first-slice").glob("*/search.json")),
    ]:
        if manifest.exists():
            exposed_ids.update(
                (t["benchmark"], t["task_id"]) for t in json.loads(manifest.read_text())["tasks"]
            )
    for row in records:
        if (row["benchmark"], row["id"]) in exposed_ids:
            row["development_only"] = True
    identities = {}
    for benchmark in BENCHMARKS:
        source_rows = [r for r in records if r["benchmark"] == benchmark]
        module = multitask_core if benchmark in multitask_core.DOMAINS else multitask_extended
        identities[benchmark] = {
            "dataset_revision": digest([r.get("source", {}) for r in source_rows]),
            "protocol_id": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            "inventory_sha256": digest(source_rows),
        }
    policy = json.loads(
        (ROOT / "submissions/superharness-astra-terra-pilot/submission.json").read_text()
    )
    result = prepare(records, identities, seed=args.seed, policy=policy)
    write_json(args.output / "summary.json", result["summary"])
    (args.output / "allocation.json.gz").write_bytes(
        gzip.compress(
            (json.dumps(result["allocation"], indent=2, allow_nan=False) + "\n").encode(), mtime=0
        )
    )
    for name, manifest in result["manifests"].items():
        write_json(args.output / "panels" / f"{name}.json", manifest)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
