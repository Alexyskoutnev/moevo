# Live task progress and reference baselines

## Overview

The default page shows one stage-progress bar, a task-status grid, the aggregate
score and improvement, and readable vector charts. Hover chart points for exact
scores. Each task tile opens the full task table; detailed protocols and candidate
history are collapsed. **Benchmarks** contains the base-Codex comparison, and
**Code & diff** contains the exact candidate source and changes.

The visible **Instruction-only test** label is deliberate: these existing runs
change shared instructions, not executable functions. They must not be relabeled
as executable harness evolution when that separate runtime is introduced.

## What the monitor counts

- **Current stage:** saved task scores for the starting agent, current quick check,
  or current full test. Zero credit still counts as a scored task.
- **Slice coverage:** distinct task IDs scored at least once anywhere in this run.
  Repeated candidates do not increase the number of independent tasks.
- **Task attempts:** executions charged before dispatch, including failures. The
  cap is a resource limit, not a promise to execute that many tasks.
- **Search steps:** four planned steps with one retry per step permit up to eight
  candidate attempts. The step bar uses completed checkpoints, not screen count.
- **Active work:** an observed experiment process plus a matching attempt directory.
  The evolution scheduler waits for both tasks in a pair before advancing. The
  reference coordinator uses a separate rolling three-worker pool per arm.
- **Last experiment activity:** result, control, or worker-log writes. Refreshing
  the browser does not reset this timestamp. Log silence alone does not prove a stall.

Tasks outside a quick check are marked **Not in this check**. A promising or audited
candidate subsequently receives the complete panel; cached quick-check results are
counted once. Only complete candidate vectors enter the scientific plots.

The current fresh-task run and earlier fixture diagnostic are standalone,
single-slice runs. Neither constitutes a completed slice of the separately
prepared eight-slice study.

## Baseline definitions

| Arm | Shared MOEvo instructions | Execution and grading |
| --- | --- | --- |
| Codex CLI task-only (Astra) | Removed from solver prompts | Account Codex CLI, Astra/xhigh, benchmark MCP tools, fixed task prompts, Terra/medium rubric judges |
| Starting harness, matched rerun | Original seed instructions retained | Same task IDs, tools, graders, settings, and corrected runtime |
| Evolved SuperHarness | Candidate instructions | Current evolution results retain their original protocol and separate history |

The original dashboard label **Base Astra** referred to the unevolved shared
instruction seed. It has been corrected to **Starting harness**. Account access
uses the Codex CLI agent, so a second run with the same instruction removal would
not measure an independent unwrapped Astra model. No such score is claimed.

The CLI reference is a controlled shared-instruction ablation with benchmark tools.
It is not a measurement of an entirely unmodified commercial CLI installation.
Judges and the simulated customer receive unchanged prompts. Actual effective
prompts, differences, call roles, model identities, and account-authentication
command checks are saved for each reference call.

## Corrected direct-answer rule

The original container runtime required at least one tool call before grading.
This rejected three successfully returned arithmetic answers during the earlier
fixture diagnostic. A separate native-grader audit found all three answers earned
full credit. Original ledger failures and incomplete candidates remain preserved;
the audit does not retroactively complete a search.

The versioned `container_runtime_v2.py` permits a successful response with zero tool
calls and sends it to the native grader. It records the actual count and does not
create a fake tool trace. Transport failures remain errors. Both reference arms use
this rule, so the matched comparison does not mix eligibility rules.

## Launch and resume

Prepare and run one FinQA case in each separate reference directory:

```sh
.venv/bin/python -m experiments.run_reference_baselines \
  --source-run results/first-slice/astra-terra-new-02 \
  --arm codex_cli_task_only \
  --output results/reference-baselines/astra-codex-task-only-01 \
  --task finqa

.venv/bin/python -m experiments.run_reference_baselines \
  --source-run results/first-slice/astra-terra-new-02 \
  --arm seed_reference \
  --output results/reference-baselines/astra-seed-reference-01 \
  --task finqa
```

Continue the fixed panel without repeating scored tasks:

```sh
.venv/bin/python -m experiments.run_reference_baselines \
  --output results/reference-baselines/astra-codex-task-only-01 --resume

.venv/bin/python -m experiments.run_reference_baselines \
  --output results/reference-baselines/astra-seed-reference-01 --resume
```

Failed attempts stay visible and are retried only with `--retry-errors`, within the
original resource limits. Changing frozen code, input identities, runtime images,
or settings requires a new run. These single-panel development observations do
not establish benchmark-wide accuracy, generalization, or uncertainty across searches.

Attach both reference directories with repeated `--reference-run` options on
`experiments.evolution_dashboard`. The **Base Codex comparison** tab rejects mismatched
task/model identities and never silently selects the best of duplicate runs.

## Parallel reference amendment

The two reference runs now use `experiments.run_reference_parallel`, with three
task workers per arm. Each takeover fenced only the verified serial coordinator;
its in-flight worker was adopted. Previously scored tasks were not repeated.
Atomic shared ledgers retain the original task/model caps and failed attempts.
Rubric grading itself uses two threads, so six task workers can create more than
six simultaneous model requests.

`run.json` and its `workers: 1` field remain part of the original frozen identity.
`execution_amendments/` records the effective concurrency, time, exact source
snapshots, preserved attempts, coordinator identities, and timing comparability.
The dashboard reads effective concurrency from this explicit amendment.

The versioned worker also repairs a preloaded judge-client alias. That alias had
bypassed the account-call audit wrapper and produced **Uncharged Codex command
blocked** before judge inference. The repair retains prompts, models and graders.
`--retry-audit-guard-errors` replaces only that identified infrastructure failure
once, with a new charged attempt; original failures remain recorded.

To resume after a coordinator exit, use the frozen parallel implementation:

```sh
.venv/bin/python -m experiments.run_reference_parallel \
  --output results/reference-baselines/astra-codex-task-only-01 \
  --workers 3 --retry-audit-guard-errors
```

An active serial coordinator additionally requires its verified `--takeover-pid`.
Never terminate a coordinator process group to perform a handoff. Unsettled model
calls block replay; a live worker retains its original wall-clock deadline.
