# Eight real development slices for the new benchmark suite

Design and local-data review, 2026-09-15. This prepares new task identities and
execution checks while the existing thirteen-fixture epoch continues. It does
not reinterpret that epoch as a full benchmark evaluation or inspect historical
paper results.

## Proposed first study

Select **ten independent task families per benchmark**, without using model
scores. Assign eight families to development slices S1–S8, one to a fixed
selection panel, and one to a reserved final panel. Select one task from each
assigned family for this bounded first study; retain every sibling's partition
assignment so later expansion cannot move a related task across partitions.

If all thirteen benchmarks supply ten suitable families and runnable tasks:

| Panel | Tasks | Use |
|---|---:|---|
| Each development slice S1–S8 | 13 | One task per benchmark; evolve on this slice |
| All eight development slices | 104 | Distinct original tasks; no repeated fixture labels |
| Fixed selection monitor | 13 | Compare each slice's chosen harness with the original seed |
| Reserved final panel | 13 | Execution blocked until freshness and final protocol are established |
| Total selected task identities | 130 | One representative from each of 130 assigned families |

This is the minimum viable eight-slice design, not a sample-size justification
for full benchmark claims. GeneBench's ten public cases constrain its allocation
to **8/1/1**, leaving one observation in each panel/slice. Expanding other
benchmarks must not duplicate those ten cases to fabricate more science tasks.

## Local availability versus runnable adapters

These counts describe local source files, before family reconciliation,
preflight exclusions, or model execution. Dataset download does not establish
that every task can run through the current adapter.

| Benchmark | Local task availability | Required grouping and current limitation |
|---|---:|---|
| FinQA | 6,251 train; 883 dev; 1,147 test | Train inventory has 741 report/year groups. Identical public input content also merges groups. Use the training pool for this provisioned study; do not mix existing public splits silently. |
| BizFinBench.v2 | 1,000 English numeric tasks; 13,310 rows across eight English categories | Only the numeric grader is connected here. Original rows lack a stable provided task ID, so preserve the recorded category/ordinal identity and pinned source revision. Exact duplicate content is grouped; absent shared-report metadata remains a grouping limitation. |
| GeneBench Pro public | 10 problem directories | Keep each original problem; complete native grading support must be checked for every selected problem. The current fixture adapter fixes `wf_selection`. |
| AMO-Bench | 50 tasks, of which 39 use the connected parser-only answer types | Eleven description-graded tasks remain outside this account/native-grader protocol. Preserve original question identities and group duplicate content. |
| PutnamBench | 672 Lean source files | The restricted proof-term adapter supports only validated single-hole statements; source-file count is not the eligible count. Group subproblems from the same competition year. Current smoke uses one fixed theorem. |
| TravelPlanner | 45 train; 180 validation with constraint fields; 1,000 test rows | Current adapter selects a valid annotated training fixture. The test CSV lacks several fields required by this local hard-constraint grading path. Begin with the supported training protocol and explicitly record annotation/preflight failures. |
| ORAgentBench | 107 task directories | Group related instance/template families, preserve their original task names, and validate each task's runtime and output schema. Only one task has been exercised by the current fixture adapter. |
| DSBench analysis | 466 questions from 38 competitions | Keep every question sharing a competition/workbook in one partition. A family cap differs from a question cap. |
| GDPval | 220 tasks | Group shared reference-material families when identifiable; preserve task IDs and source/file hashes. The existing adapter selects the first artifact task. |
| Terminal-Bench 2 | 89 task directories | Native runtime/test compatibility is task-specific. The current generic mini adapter supports `regex-log`; that does not make the other 88 tasks runnable. |
| Harvey LAB | 2,010 task configurations across 27 legal areas | Group scenario/document siblings, not an entire broad legal area. Check document availability, extraction, deliverable names, and criterion grading per selected task. |
| HealthBench Professional | 525 cases | Group duplicate/shared conversations; retain use case, specialty, and difficulty as metadata. Hide physician responses and rubrics from the solver. |
| τ³-bench retail | 114 tasks: 74 native train and 40 native test | Preserve native task IDs/splits and group shared customer/order/scenario families when identifiable. Other local simulator domains are separate protocol expansions, not extra retail tasks. |

The six new core inventory adapters report 7,811 task records across FinQA,
BizFin numeric, GeneBench, AMO parser, TravelPlanner train, and DSBench analysis.
FinQA contains 57 duplicate public-input entries; the splitter merges identical
content rather than counting renamed duplicates as independent families.

The extended inventory currently passes structural checks for 208/220 GDPval
tasks, 2,010/2,010 Harvey tasks, 525/525 HealthBench cases, 326/672 Putnam
statements, 99/107 ORAgentBench tasks, 73/114 retail simulator tasks, and
**1/89 Terminal-Bench tasks**. Their eligible family counts are respectively
207, 1,493, 525, 63, 87, 53, and **one**. These are static adapter/schema/asset
checks, not fresh end-to-end runs. Terminal-Bench therefore currently blocks an
all-thirteen eight-slice execution. The existing `regex-log` fixture must not be
copied into eight nominally different tasks.

A one-slice bridge with the twelve other benchmarks and the known
`regex-log` fixture can test newly selected task adapters. It needs its own
protocol label and exact IDs; it does not establish the thirteen-benchmark
eight-slice allocation or a fresh terminal-task comparison.

At least ten independent groups are necessary for each included benchmark.
If grouping, task restrictions, or runtime checks leave fewer than ten, record
the benchmark as excluded/blocked for this eight-slice study. Do not fill the
gap by assigning the same task new IDs. A partially prepared suite must list
its excluded benchmarks explicitly.

## Splitter and reproducibility contract

The implementation is `moevo/data/splitters/multidomain.py`:

```python
manifest = build_multidomain_slices(
    records,
    seed=20260915,
    benchmark_identities={
        "benchmark_name": {
            "dataset_revision": "pinned source revision",
            "protocol_id": "pinned adapter/grader/runtime identity",
        },
    },
    expected_benchmarks=[...],
    development_slices=8,
    max_groups_per_benchmark=10,
    final_policy="reserve_only",
)
```

Each record has original `id`, `benchmark`, `domain`, `group_id`,
`content_sha256`, and explicit `exposed`. The optional boolean
`development_only` marks a known prior fixture. Additional source and readiness
metadata is preserved in the input fingerprint. Task references are
`{benchmark, id}`, so IDs in different datasets cannot collide.

The splitter merges declared families and exact duplicate content transitively
within a benchmark. It conservatively excludes benchmarks with overlapping
content across benchmark boundaries until that overlap is reconciled. Whole
components receive one partition. Input order cannot change assignments;
record contents, source/protocol identities, exposure flags, seed, and allocation
settings have hashes in the manifest.

Duplicate detection is only as complete as the inventory's hash definition and
family metadata. The extended adapter currently hashes the full task/grading
record with provenance, while separately hashing input assets at preflight.
Different IDs or rubrics can therefore hide identical public inputs from this
hash-based check. Its explicit family grouping still applies, but a future
freshness claim needs a separate public-input fingerprint and reconciliation;
the current reservation makes no such claim.

`component_assignments` exposes each component's records and partition. The
provisioner can select a deterministic representative while keeping all sibling
assignments. `max_groups_per_benchmark` caps families, not tasks; available,
selected, unused, and per-partition counts remain separate. Changing an ID,
content hash, exposure status, or protocol changes the manifest identity.

Strict `require_unexposed` mode keeps exposed families in development and needs
clean selection/final families. For this provisioning pass, unknown previous
exposure is conservatively `true`, and `reserve_only` allows disjoint development
preparation while keeping **`final_evaluation_allowed=false`**. A random new
partition does not make previously exposed data fresh. Dataset and adapter
readiness are separate flags; allocation alone does not pass native preflight.
Known prior fixtures have `development_only=true`: their complete family and
duplicate-content component can enter only development or the unused pool,
even in reserve-only mode. Insufficient eligible families produce an explicit
exclusion instead of relaxing this restriction to fill a reserved panel.

## Eight-slice execution sequence

For each of S1–S8:

1. Load the harness carried from the preceding slice; S1 starts with the
   original unevolved seed. Record the incoming source hash and predecessor.
2. Evaluate the incoming harness on the new slice. Its previous slice's scores
   must not become scores on these new tasks.
3. Evaluate the original unevolved seed on the same slice for a paired starting
   comparison; reuse the incoming evaluation only if the artifact is identical.
4. Perform four mutation steps with paired screens. The last screening step
   receives full evaluation even without a screen improvement. Admit only
   candidates with complete vectors on the current slice's thirteen tasks.
5. Choose one actual harness using the declared rule and carry its code to the
   next slice. Record failed attempts and ties; improvement is not required.
6. Evaluate that chosen harness on the fixed selection panel, alongside the
   original seed baseline. Keep this panel's feedback out of mutation prompts.

The fixed panel is a **selection/development monitor**. It remains development
evidence even if only the user sees its aggregate curves: the user can make
subsequent tuning decisions from them. Final evaluation must use a separate
launch after the selected harness and all settings are frozen; it remains
blocked in this provisioned study.

Four proposals with a fresh 24-slot/two-island NSGA-III population cannot
exercise over-capacity survival within a slice. This run validates the cascade,
task execution, mutation, screening, and carry-forward connections. Native
selection behavior has offline tests; algorithm-comparison evidence requires a
separate design with independent complete search seeds and enough candidates.

## Budget and completion checks

The proposed upper bound is **728 task attempts**:

- Eight slices × 65 evolution task attempts = 520.
- Seven additional slices × 13 original-seed checks = 91; S1 reuses its identical
  incoming seed evaluation.
- One 13-task selection-seed evaluation plus eight 13-task chosen-harness
  evaluations = 117.

There are **32 mutation steps**, not 32 epochs. The eight-slice launcher disables
proposal retries and caps proposal calls at 32. The separate one-slice bridge
allows one retry per step, so its four steps can use up to eight proposal calls.
Exact caches can lower physical execution
counts. Task budgets exclude repeated grader controls and may include multiple
solver, simulator, and criterion-judge calls per task; track those calls and
tokens separately. Estimate runtime from the actual selected tasks and judge
workload before launching the full study.

Meaningful smoke tests should check:

- Every selected task ID exists and has a native adapter/runtime preflight;
  thirteen successful old fixtures do not satisfy this for new IDs.
- All eight slices, selection, and reserved final are disjoint by identity,
  family component, and duplicate content; exposed final records block final use.
- The predecessor's chosen source hash equals the next slice's incoming hash,
  while scoring uses that next slice's task identities.
- Original-seed comparisons use the original source, and aggregate vectors
  reconstruct from the same complete task panel and declared metric mapping.
- Checkpoints preserve selected tasks, source/config identities, pending work,
  usage charges, and lineage; resume cannot silently change a slice or protocol.
- Each claimed mutation has changed instructions and real candidate-output
  grading. A completed process alone does not prove the evolution path ran.
- Selection monitoring never contributes task answers or criterion feedback to
  the mutation prompt, and reserved final tasks are never executed by this launch.

The study is complete when its declared task/mutation paths finish or report
explicit terminal failures and the records reconcile. A gain is not a passing
condition. Aggregate percentages should identify the panel, task count, native
metric, and weighting rule. Domain means and benchmark means differ because
finance and mathematics each contain two benchmarks. This run can reveal
development trends and runtime problems; it cannot establish saturation, SOTA,
or final-test generalization from one task per benchmark per slice.
