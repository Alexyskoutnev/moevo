# Live task progress and reference baselines

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
  The queue never assumes a free worker starts another task: the current scheduler
  waits for both tasks in a pair before advancing.
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
