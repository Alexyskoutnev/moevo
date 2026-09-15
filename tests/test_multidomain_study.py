"""Study preparation preserves real identity and never authorizes an unverified final set."""

import hashlib

from experiments.prepare_multidomain_study import BENCHMARKS, prepare


def inventory():
    records = []
    for benchmark in BENCHMARKS:
        for group in range(10):
            # Two sibling tasks per report: only one is used, both remain in its partition.
            for sibling in range(2):
                task_id = f"original-{group}-{sibling}"
                records.append(
                    {
                        "id": task_id,
                        "benchmark": benchmark,
                        "domain": benchmark,
                        "group_id": f"report-{group}",
                        "content_sha256": hashlib.sha256(
                            f"{benchmark}/{task_id}".encode()
                        ).hexdigest(),
                        "exposed": True,
                        "adapter_ready": True,
                    }
                )
    identities = {
        b: {"dataset_revision": "test-data", "protocol_id": "test-grader"} for b in BENCHMARKS
    }
    return records, identities


def test_eight_panels_use_disjoint_families_and_130_original_tasks():
    records, identities = inventory()
    result = prepare(records, identities, seed=42, policy={})
    assert result == prepare(list(reversed(records)), identities, seed=42, policy={})
    panels = result["manifests"]
    assert len(panels) == 10
    assert all(len(panel["tasks"]) == 13 for panel in panels.values())
    tasks = [t for panel in panels.values() for t in panel["tasks"]]
    assert len({(t["benchmark"], t["task_id"]) for t in tasks}) == 130
    assert len({t["component"] for t in tasks}) == 130
    assert result["summary"]["task_attempt_caps"]["total"] == 728
    assert result["summary"]["status"] == "prepared"
    assert result["summary"]["final_evaluation_allowed"] is False
    assert not result["summary"]["new_tasks_e2e_validated"]
    assert panels["selection"]["split"] == "selection"
    assert panels["final"]["split"] == "final"


def test_downloaded_but_unsupported_tasks_block_run_readiness():
    records, identities = inventory()
    for row in records:
        if row["benchmark"] == "terminal_bench_2":
            row["adapter_ready"] = False
    result = prepare(records, identities, seed=42, policy={})
    assert result["summary"]["status"] == "preparation_incomplete"
    assert len(result["summary"]["unsupported_tasks"]) == 9
    assert result["summary"]["missing_benchmarks"] == {}


def test_insufficient_dataset_is_explicit_and_never_repeated():
    records, identities = inventory()
    records = [
        r for r in records if r["benchmark"] != "genebench_pro" or r["group_id"] == "report-0"
    ]
    result = prepare(records, identities, seed=42, policy={})
    assert all(
        "genebench_pro" in missing for missing in result["summary"]["missing_benchmarks"].values()
    )
    assert result["summary"]["status"] == "preparation_incomplete"
