"""Split identities and exposure invariants, without inference or dataset downloads."""

from __future__ import annotations

import copy
import hashlib
import json
import random

import pytest

from moevo.data.splitters.multidomain import build_multidomain_slices


def inventory(benchmark="alpha", groups=10, group_size=1, exposed=0):
    return [
        {
            "id": f"task-{group:03d}-{part}",
            "benchmark": benchmark,
            "domain": "science",
            "group_id": f"family-{group:03d}",
            "content_sha256": hashlib.sha256(f"{benchmark}/{group}/{part}".encode()).hexdigest(),
            "exposed": group < exposed,
            "adapter_ready": True,
            "source": {"revision": "data-r1", "original_id": f"task-{group:03d}-{part}"},
        }
        for group in range(groups)
        for part in range(group_size)
    ]


def build(rows, **kwargs):
    identities = {
        b: {"dataset_revision": "data-r1", "protocol_id": "grader-r1"}
        for b in {r["benchmark"] for r in rows}
    }
    return build_multidomain_slices(rows, seed=20260915, benchmark_identities=identities, **kwargs)


def assignments(manifest):
    parts = {
        **manifest["development_slices"],
        "selection": manifest["selection_ids"],
        "final": manifest["final_ids"],
    }
    result = {}
    for partition, refs in parts.items():
        for ref in refs:
            identity = (ref["benchmark"], ref["id"])
            assert identity not in result
            result[identity] = partition
    return result


def test_thirteen_benchmarks_have_eight_real_slices_and_two_disjoint_reserves():
    rows = [row for i in range(13) for row in inventory(f"benchmark-{i}", exposed=1)]
    manifest = build(rows)
    assert all(len(part) == 13 for part in manifest["development_slices"].values())
    assert len(manifest["selection_ids"]) == len(manifest["final_ids"]) == 13
    assigned = assignments(manifest)
    assert len(assigned) == 130
    assert set(assigned) == {(r["benchmark"], r["id"]) for r in rows}
    assert manifest["development_execution_ready"]
    assert manifest["final_evaluation_allowed"]
    assert all(assigned[(r["benchmark"], r["id"])].startswith("S") for r in rows if r["exposed"])


def test_family_members_and_content_duplicates_never_cross_partitions():
    rows = inventory(groups=12, group_size=3)
    # Two different declared families contain an exact duplicate; they form a
    # single larger component, including every member of both families.
    rows[3]["content_sha256"] = rows[0]["content_sha256"]
    rows[1]["exposed"] = True
    manifest = build(rows)
    assigned = assignments(manifest)
    assert manifest["benchmarks"]["alpha"]["available_groups"] == 11
    for group in range(12):
        roles = {
            assigned[("alpha", r["id"])] for r in rows if r["group_id"] == f"family-{group:03d}"
        }
        assert len(roles) == 1
    duplicate_roles = {assigned[("alpha", r["id"])] for r in rows[:6]}
    assert len(duplicate_roles) == 1
    assert next(iter(duplicate_roles)).startswith("S")


def test_seed_hashes_input_order_and_benchmark_additions_are_reproducible():
    rows = inventory(groups=40, group_size=2, exposed=3)
    first = build(rows)
    shuffled = copy.deepcopy(rows)
    random.Random(90).shuffle(shuffled)
    assert build(shuffled) == first
    more = build(rows + inventory("beta", groups=40))
    assert {k: v for k, v in assignments(more).items() if k[0] == "alpha"} == assignments(first)
    identities = {"alpha": {"dataset_revision": "data-r1", "protocol_id": "grader-r1"}}
    changed_seed = build_multidomain_slices(rows, seed=999, benchmark_identities=identities)
    assert assignments(changed_seed) != assignments(first)
    assert changed_seed["manifest_sha256"] != first["manifest_sha256"]
    identities["alpha"]["protocol_id"] = "grader-r2"
    changed_protocol = build_multidomain_slices(
        rows, seed=20260915, benchmark_identities=identities
    )
    assert changed_protocol["input_sha256"] != first["input_sha256"]
    assert changed_protocol["manifest_sha256"] != first["manifest_sha256"]
    assert assignments(changed_protocol) == assignments(first)
    assert json.loads(json.dumps(first, allow_nan=False)) == first


def test_too_few_independent_groups_are_excluded_without_relabeling_or_duplication():
    rows = inventory(groups=9, group_size=20) + inventory("ready")
    manifest = build(rows, expected_benchmarks=["alpha", "ready", "missing"])
    assert manifest["benchmarks"]["alpha"]["status"] == "excluded"
    assert manifest["benchmarks"]["missing"]["reason"] == "no_records"
    assert manifest["benchmarks"]["ready"]["status"] == "included"
    assert not manifest["development_partition_ready"]
    assert not manifest["final_evaluation_allowed"]
    assert set(b for b, _ in assignments(manifest)) == {"ready"}
    assert len(manifest["unused_ids"]) == 180


def test_unknown_exposure_can_provision_reserves_but_never_fresh_final_results():
    rows = inventory(groups=10, exposed=10)
    strict = build(rows)
    assert strict["benchmarks"]["alpha"]["status"] == "excluded"
    reserved = build(rows, final_policy="reserve_only")
    assert reserved["development_partition_ready"]
    assert reserved["development_execution_ready"]
    assert not reserved["final_freshness_verified"]
    assert not reserved["final_evaluation_allowed"]
    assert len(assignments(reserved)) == len(rows)
    assert all(len(s) == 1 for s in reserved["development_slices"].values())


def test_fresh_reserve_only_still_requires_explicit_release_before_final_execution():
    manifest = build(inventory(), final_policy="reserve_only")
    assert manifest["final_freshness_verified"]
    assert not manifest["final_evaluation_allowed"]


def test_known_fixtures_and_related_tasks_stay_development_only_in_reserve_mode():
    rows = inventory(groups=12, group_size=2, exposed=12)
    rows[0]["development_only"] = True
    # One known fixture protects its sibling and another otherwise distinct
    # family linked by a duplicate input, even when all exposure is unknown.
    rows[2]["content_sha256"] = rows[0]["content_sha256"]
    manifest = build(rows, final_policy="reserve_only", max_groups_per_benchmark=10)
    assigned = assignments(manifest)
    assert manifest["benchmarks"]["alpha"]["development_only_groups"] == 1
    assert all(assigned[(r["benchmark"], r["id"])].startswith("S") for r in rows[:4])
    assert not manifest["final_evaluation_allowed"]


def test_known_fixture_restrictions_cannot_be_relaxed_to_fill_reserves():
    rows = inventory(groups=10, exposed=10)
    for row in rows[:9]:
        row["development_only"] = True
    manifest = build(rows, final_policy="reserve_only")
    assert manifest["benchmarks"]["alpha"]["status"] == "excluded"
    assert manifest["benchmarks"]["alpha"]["reason"] == (
        "need_two_groups_not_restricted_to_development"
    )
    assert not manifest["development_partition_ready"]
    assert not manifest["selection_ids"] and not manifest["final_ids"]
    assert len(manifest["unused_ids"]) == 10


def test_group_cap_preserves_whole_families_and_reports_task_counts_separately():
    rows = inventory(groups=30, group_size=4, exposed=20)
    manifest = build(rows, max_groups_per_benchmark=10)
    summary = manifest["benchmarks"]["alpha"]
    assert summary["available_tasks"] == 120
    assert summary["available_groups"] == 30
    assert summary["selected_groups"] == 10
    assert summary["selected_tasks"] == 40
    assert len(manifest["unused_ids"]) == 80
    assigned = assignments(manifest)
    assert len(assigned) == 40
    assert all(
        assigned[(r["benchmark"], r["id"])].startswith("S")
        for r in rows
        if r["exposed"] and (r["benchmark"], r["id"]) in assigned
    )


def test_cross_benchmark_content_overlap_is_explicitly_excluded():
    rows = inventory() + inventory("beta") + inventory("clean")
    rows[10]["content_sha256"] = rows[0]["content_sha256"]
    manifest = build(rows)
    assert {row["benchmark"] for row in manifest["excluded_benchmarks"]} == {"alpha", "beta"}
    assert set(b for b, _ in assignments(manifest)) == {"clean"}
    assert not manifest["development_partition_ready"]


def test_allocated_data_does_not_silently_make_unimplemented_adapters_ready():
    rows = inventory()
    rows[0]["adapter_ready"] = False
    manifest = build(rows)
    assert manifest["development_partition_ready"]
    assert not manifest["development_execution_ready"]
    assert not manifest["final_evaluation_allowed"]
    assert manifest["benchmarks"]["alpha"]["adapter_ready_tasks"] == 9
    rows[0].pop("adapter_ready")
    assert not build(rows)["development_execution_ready"]


@pytest.mark.parametrize(
    "change",
    [
        {"exposed": None},
        {"development_only": "true"},
        {"content_sha256": "not-a-hash"},
        {"group_id": ""},
        {"id": ""},
    ],
)
def test_malformed_records_are_rejected(change):
    rows = inventory()
    rows[0].update(change)
    with pytest.raises(ValueError):
        build(rows)


def test_bad_identity_configuration_and_impossible_caps_are_rejected():
    rows = inventory()
    with pytest.raises(ValueError, match="Duplicate"):
        build([*rows, rows[0]])
    with pytest.raises(ValueError, match="pin"):
        build_multidomain_slices(rows, seed=42, benchmark_identities={})
    with pytest.raises(ValueError, match="at least 10"):
        build(rows, max_groups_per_benchmark=9)
    with pytest.raises(ValueError, match="proportions"):
        build(rows, proportions=(0.8, 0.2, 0))
    rows[0]["domain"] = "different"
    with pytest.raises(ValueError, match="one declared domain"):
        build(rows)
