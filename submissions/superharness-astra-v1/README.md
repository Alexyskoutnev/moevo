# Real SuperHarness mini run

This is a **real, unevolved integration run**, using one frozen instruction policy
and one task per benchmark with a working runtime. It is not a leaderboard
submission or a full benchmark baseline.

See [results and qualifications](../../docs/mini-run-status.md) and
[the metadata-only report](validation.json). A failed infrastructure check has a
null score. A valid wrong model answer can have score zero.

## Run on the prepared checkout

From the repository root:

```sh
uv sync --group dev --group benchmarks
codex login status
.venv/bin/python experiments/run_mini_suite.py --output results/mini-suite/new-run
.venv/bin/python experiments/report_mini_suite.py results/mini-suite/new-run
```

`codex login status` must report a ChatGPT login. The runner forces account
authentication, ignores user model/provider settings, strips API-key environment
variables, and disables inference API fallback. Model and judge calls use
`gpt-6-astra` at `xhigh`. The older API-based evaluators remain available for
historical experiments but are not called by this mini-suite.

Each solver has a 1,200-second limit and, where applicable, 80 permitted task
tool calls. Graders have separate limits. τ³ uses its native simulator with a
50-step cap and account-Astra user simulation. Docker tasks run offline with
2 CPUs and 4 GB memory; these resources differ from some official protocols.
Official source graders and task inputs are separated from the agent's workspace.

Use `--benchmarks finqa gdpval ...` to select entries. Re-running the same output
directory resumes failed/setup-blocked entries and keeps completed entries.
Use a fresh directory to run another model attempt. Changing the policy requires
a new directory. Grader-failed controls block scoring rather than producing zero.

The recorded run used two sequentially started shards, with two workers each:

```sh
.venv/bin/python experiments/run_mini_suite.py \
  --benchmarks finqa amo bizfinbench2 genebench_pro dsbench healthbench_professional travelplanner automationbench
.venv/bin/python experiments/run_mini_suite.py \
  --output results/mini-suite/astra-v1-extended \
  --benchmarks oragentbench terminal_bench_2 gdpval harvey_lab tau3_bench terminal_bench_science putnambench eduagentbench
.venv/bin/python experiments/audit_automation.py \
  results/mini-suite/astra-v1/automationbench/attempt-01
.venv/bin/python experiments/report_mini_suite.py \
  results/mini-suite/astra-v1 results/mini-suite/astra-v1-extended
```

## Runtime setup

Downloaded data and upstream repositories are not committed. Source manifests in
`validation.json` record the revisions used. Exact image IDs for this run are in
[runtime-images.json](runtime-images.json). The preparers
`experiments/prepare_domain_data.py` and `experiments/prepare_domain_sources.py`
download public assets and record hashes. Existing manifests retain their pinned
versions; an empty checkout discovers current upstream versions, so compare and
restore the recorded revisions before an exact reproduction. DSBench and
TravelPlanner additionally require their public asset archives.

Build the local runtimes after preparing the corresponding sources:

```sh
docker build -t moevo-or-runtime:20260915 -f experiments/runtime/Dockerfile .
docker build -t moevo-doc-runtime:20260915 -f experiments/runtime/Dockerfile.documents .
docker build -t moevo-terminal-verifier:20260915 -f experiments/runtime/Dockerfile.terminal-verifier .
docker build -t moevo-putnam-runtime:20260915 -f experiments/runtime/Dockerfile.lean data/external/putnambench/lean4
docker build -t moevo-cilia-agent:20260915 data/external/terminal_bench_science/tasks/life-sciences/biology/cilia-segmentation/environment
docker build -t moevo-cilia-verifier:20260915 data/external/terminal_bench_science/tasks/life-sciences/biology/cilia-segmentation/tests
```

The science agent and verifier use separate official images. The verifier's hidden
labels are never mounted into the agent. PutnamBench uses Lean 4.27.0 and its
upstream pinned Mathlib, accepts a restricted proof term for the unchanged theorem,
and rejects admitted proofs and unapproved axioms. All generated Lean code is
compiled in Docker.

τ³ uses a separate Python 3.12 environment under `data/external/tau3_bench/.venv`.
Install the upstream frozen dependencies, `websockets==15.0.1` (needed by an
upstream unconditional import), and this checkout as an editable package. The
resulting environment differs from the untouched upstream lock; exact installed
package versions for both host and τ³ are in [runtime-packages.json](runtime-packages.json).

GDPval evaluates extracted deliverable content using the public weighted rubric;
it does not reproduce the official expert pairwise evaluation or visual review.
Harvey LAB uses the released criterion prompt and all-pass rule with a single
account-Astra judge, replacing its standard two-model judge configuration.
These variants are explicit in the run report.

## Evidence and limits

Local `results/mini-suite/` contains real model transcripts, permitted tool traces,
submitted artifacts, grader outputs, controls, and attempt directories. This
directory and `data/` remain ignored by Git. The published JSON contains metadata,
scores, hashes, and qualifications without benchmark answers or credentials.

The AutomationBench case has a confirmed task-ID mismatch; its unmodified native
score stays under audit. EduAgentBench's downloaded public release contains tasks
and mock course assets but no evaluator or Canvas runtime, so it cannot yet be
counted as an official end-to-end integration.

Single-task scores cannot establish saturation, useful evolutionary headroom,
or state-of-the-art performance. A separate frozen development/selection/test
protocol is required before comparing evolved harnesses.
