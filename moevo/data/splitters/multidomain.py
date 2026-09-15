"""Group-aware, deterministic multi-benchmark development/selection/final splits.

This module allocates identities; it does not load benchmarks or execute models.
Eight development slices plus selection and final need at least ten independent
groups per benchmark. Small public sets are never repeated to meet that minimum.
Unknown previous exposure must be supplied as ``exposed=True``. Reserve-only
allocation can provision development while explicitly blocking final evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

_REQUIRED = ("id", "benchmark", "domain", "group_id", "content_sha256")
_HASH = re.compile(r"[0-9a-fA-F]{64}\Z")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _ref(record: Mapping[str, Any]) -> dict[str, str]:
    return {"benchmark": record["benchmark"], "id": record["id"]}


def _components(records: list[dict]) -> list[dict]:
    """Merge declared families and identical content transitively within a benchmark."""
    parents = list(range(len(records)))

    def root(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen: dict[tuple[str, str], int] = {}
    for i, record in enumerate(records):
        for kind in ("group_id", "content_sha256"):
            key = (kind, record[kind])
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    groups: dict[int, list[dict]] = defaultdict(list)
    for i, record in enumerate(records):
        groups[root(i)].append(record)
    return [
        {
            "key": _digest([_ref(r) for r in group]),
            "records": group,
            "exposed": any(r["exposed"] for r in group),
            "development_only": any(r.get("development_only") is True for r in group),
        }
        for group in groups.values()
    ]


def _allocation_counts(
    count: int, proportions: Sequence[float], development_slices: int, exposed: int
) -> dict[str, int]:
    # Minimum coverage takes priority; the remaining groups move toward the
    # requested proportions. Exposure can require a larger development share.
    allocated = [max(development_slices, exposed), 1, 1]
    if sum(allocated) > count:
        raise ValueError("Not enough groups for the minimum partition coverage")
    targets = [p * count for p in proportions]
    for _ in range(count - sum(allocated)):
        selected = max(range(3), key=lambda i: (targets[i] - allocated[i], -i))
        allocated[selected] += 1
    return dict(zip(("development", "selection", "final"), allocated, strict=True))


def build_multidomain_slices(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    benchmark_identities: Mapping[str, Mapping[str, Any]],
    development_slices: int = 8,
    proportions: Sequence[float] = (0.8, 0.1, 0.1),
    expected_benchmarks: Sequence[str] | None = None,
    max_groups_per_benchmark: int | None = None,
    final_policy: str = "require_unexposed",
) -> dict:
    """Allocate original record IDs and return a JSON-serializable manifest.

    Required record fields: id, benchmark, domain, group_id, content_sha256,
    exposed (explicit bool). Additional JSON metadata is preserved and hashed.
    Hashes must represent task content/inputs, not just IDs, to detect duplicates.
    Benchmark identities require dataset_revision and protocol_id; grouping and
    runtime identities can be additional fields. IDs are benchmark-local.

    Strict mode assigns every selected exposed family to development. Selection
    and final families are unexposed. Reserve-only mode permits exposed families
    in reserved partitions but never authorizes final evaluation. Missing data,
    too few groups, and cross-benchmark content overlap are explicitly excluded.
    Mark known prior fixtures ``development_only=True``; their entire families
    stay in development (or unused), including under reserve-only allocation.

    A group cap is not a task cap: all records of a selected family stay together.
    ``adapter_ready`` is only metadata; missing/false readiness prevents declaring
    the allocated study executable. This function cannot validate task runtimes.
    """
    if type(seed) is not int or type(development_slices) is not int or development_slices < 1:
        raise ValueError("Seed must be an integer and development_slices a positive integer")
    if final_policy not in {"require_unexposed", "reserve_only"}:
        raise ValueError("Unknown final_policy")
    minimum = development_slices + 2
    if max_groups_per_benchmark is not None and (
        type(max_groups_per_benchmark) is not int or max_groups_per_benchmark < minimum
    ):
        raise ValueError(f"A group cap must allow at least {minimum} independent groups")
    if (
        len(proportions) != 3
        or any(
            isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0
            for p in proportions
        )
        or not math.isclose(sum(proportions), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("Positive development/selection/final proportions must sum to one")

    normalized = []
    identities_seen = set()
    by_benchmark: dict[str, list[dict]] = defaultdict(list)
    content_benchmarks: dict[str, set[str]] = defaultdict(set)
    for raw in records:
        if not isinstance(raw, Mapping) or any(
            not isinstance(raw.get(key), str) or not raw[key].strip() for key in _REQUIRED
        ):
            raise ValueError("Every record needs nonempty ID, benchmark, domain, group and hash")
        if type(raw.get("exposed")) is not bool:
            raise ValueError("Exposure must be explicit; conservatively mark unknown exposure true")
        if "development_only" in raw and type(raw["development_only"]) is not bool:
            raise ValueError("development_only must be a boolean when supplied")
        if not _HASH.fullmatch(raw["content_sha256"]):
            raise ValueError("content_sha256 must be a 64-digit SHA-256 hex digest")
        record = {**raw, "content_sha256": raw["content_sha256"].lower()}
        _digest(record)  # Fail before assigning nonserializable or nonfinite provenance.
        identity = (record["benchmark"], record["id"])
        if identity in identities_seen:
            raise ValueError(f"Duplicate benchmark/task identity: {identity}")
        identities_seen.add(identity)
        normalized.append(record)
        by_benchmark[record["benchmark"]].append(record)
        content_benchmarks[record["content_sha256"]].add(record["benchmark"])
    normalized.sort(key=lambda r: (r["benchmark"], r["id"]))
    requested = list(expected_benchmarks if expected_benchmarks is not None else by_benchmark)
    if (
        any(not isinstance(b, str) or not b for b in requested)
        or len(set(requested)) != len(requested)
        or set(by_benchmark) - set(requested)
    ):
        raise ValueError("Expected benchmarks must be unique names covering every input record")
    requested.sort()
    frozen_identities = {}
    for benchmark in requested:
        identity = benchmark_identities.get(benchmark, {})
        if benchmark in by_benchmark and any(
            not isinstance(identity.get(key), str) or not identity[key].strip()
            for key in ("dataset_revision", "protocol_id")
        ):
            raise ValueError(f"{benchmark}: pin dataset_revision and protocol_id")
        frozen_identities[benchmark] = dict(identity)
    _digest(frozen_identities)
    overlaps = set().union(*(v for v in content_benchmarks.values() if len(v) > 1), set())

    slices: dict[str, list[dict]] = {f"S{i + 1}": [] for i in range(development_slices)}
    selection, final, unused, summaries = [], [], [], {}
    assignments = []
    for benchmark in requested:
        available = sorted(by_benchmark.get(benchmark, []), key=lambda r: r["id"])
        domains = {r["domain"] for r in available}
        if len(domains) > 1:
            raise ValueError(f"{benchmark}: one benchmark must have one declared domain")
        groups = _components(available)
        clean = [g for g in groups if not g["exposed"]]
        exposed = [g for g in groups if g["exposed"]]
        protected = [
            g
            for g in groups
            if g["development_only"] or (final_policy == "require_unexposed" and g["exposed"])
        ]
        protected_keys = {g["key"] for g in protected}
        eligible = [g for g in groups if g["key"] not in protected_keys]

        def order(group: dict, role: str = "groups", benchmark_key: str = benchmark) -> str:
            return _digest([seed, benchmark_key, role, group["key"]])

        summary = {
            "domain": next(iter(domains), None),
            "available_tasks": len(available),
            "available_groups": len(groups),
            "exposed_groups": len(exposed),
            "unexposed_groups": len(clean),
            "development_only_groups": sum(g["development_only"] for g in groups),
            "status": "excluded",
        }
        reason = None
        if not available:
            reason = "no_records"
        elif benchmark in overlaps:
            reason = "cross_benchmark_duplicate_content_requires_reconciliation"
        elif len(groups) < minimum:
            reason = f"need_{minimum}_independent_groups_for_{development_slices}_slices_plus_selection_final"
        elif final_policy == "require_unexposed" and len(clean) < 2:
            reason = "need_two_unexposed_groups_for_selection_and_final"
        elif len(eligible) < 2:
            reason = "need_two_groups_not_restricted_to_development"
        if reason:
            summaries[benchmark] = {**summary, "reason": reason}
            unused.extend(_ref(r) for r in available)
            continue

        cap = min(max_groups_per_benchmark or len(groups), len(groups))
        selected_protected = sorted(protected, key=order)[: min(len(protected), cap - 2)]
        selected_eligible = sorted(eligible, key=order)[: cap - len(selected_protected)]
        chosen = selected_protected + selected_eligible
        chosen_keys = {g["key"] for g in chosen}
        unused.extend(_ref(r) for g in groups if g["key"] not in chosen_keys for r in g["records"])
        counts = _allocation_counts(
            cap,
            proportions,
            development_slices,
            len(selected_protected),
        )
        # Prefer fresh final and selection groups even in reserve-only mode.
        ranked = sorted(selected_eligible, key=lambda g: (g["exposed"], order(g, "final")))
        final_groups = ranked[: counts["final"]]
        final_keys = {g["key"] for g in final_groups}
        remaining = [g for g in chosen if g["key"] not in final_keys]
        ranked = sorted(
            [g for g in remaining if g["key"] not in protected_keys],
            key=lambda g: (g["exposed"], order(g, "selection")),
        )
        selection_groups = ranked[: counts["selection"]]
        selection_keys = {g["key"] for g in selection_groups}
        dev_groups = [g for g in remaining if g["key"] not in selection_keys]

        local_sizes = {name: 0 for name in slices}
        local_groups = {name: 0 for name in slices}
        for group in sorted(dev_groups, key=lambda g: (-len(g["records"]), order(g, "slices"))):
            name = min(slices, key=lambda s: (local_sizes[s], local_groups[s], s))
            refs = [_ref(r) for r in group["records"]]
            slices[name].extend(refs)
            local_sizes[name] += len(refs)
            local_groups[name] += 1
            assignments.append({"component": group["key"], "partition": name, "records": refs})
        for role, targets, allocated in (
            ("selection", selection, selection_groups),
            ("final", final, final_groups),
        ):
            for group in allocated:
                refs = [_ref(r) for r in group["records"]]
                targets.extend(refs)
                assignments.append({"component": group["key"], "partition": role, "records": refs})
        selected_rows = [r for g in chosen for r in g["records"]]
        fresh = not any(g["exposed"] for g in final_groups)
        summaries[benchmark] = {
            **summary,
            "status": "included",
            "selected_tasks": len(selected_rows),
            "selected_groups": len(chosen),
            "group_counts": counts,
            "development_slice_tasks": local_sizes,
            "development_slice_groups": local_groups,
            "selection_tasks": sum(len(g["records"]) for g in selection_groups),
            "final_tasks": sum(len(g["records"]) for g in final_groups),
            "final_freshness_verified": fresh,
            "all_selected_adapters_ready": all(
                r.get("adapter_ready") is True for r in selected_rows
            ),
            "adapter_ready_tasks": sum(r.get("adapter_ready") is True for r in selected_rows),
        }
    for partition in [*slices.values(), selection, final, unused]:
        partition.sort(key=lambda r: (r["benchmark"], r["id"]))
    included = [b for b in requested if summaries[b]["status"] == "included"]
    excluded = [
        {"benchmark": b, "reason": summaries[b]["reason"]} for b in requested if b not in included
    ]
    partition_ready = bool(requested) and not excluded and all(slices.values())
    adapters_ready = bool(included) and all(
        summaries[b]["all_selected_adapters_ready"] for b in included
    )
    fresh = bool(included) and all(summaries[b]["final_freshness_verified"] for b in included)
    settings = {
        "seed": seed,
        "development_slice_count": development_slices,
        "proportions": list(proportions),
        "max_groups_per_benchmark": max_groups_per_benchmark,
        "final_policy": final_policy,
        "expected_benchmarks": requested,
    }
    output = {
        "schema_version": 1,
        "splitter": "moevo_multidomain_groups_v1",
        **settings,
        "benchmark_identities": frozen_identities,
        "input_sha256": _digest({"records": normalized, "identities": frozen_identities}),
        "development_slices": slices,
        "selection_ids": selection,
        "final_ids": final,
        "unused_ids": unused,
        "benchmarks": summaries,
        "excluded_benchmarks": excluded,
        "component_assignments": sorted(assignments, key=lambda a: a["component"]),
        "records": normalized,
        "development_partition_ready": partition_ready,
        "development_execution_ready": partition_ready and adapters_ready,
        "final_freshness_verified": partition_ready and fresh,
        "final_evaluation_allowed": partition_ready
        and adapters_ready
        and fresh
        and final_policy == "require_unexposed",
        "readiness_note": "Allocation is not E2E validation. Final use also requires frozen selection and executor approval of the protocol.",
    }
    output["manifest_sha256"] = _digest(output)
    return output
