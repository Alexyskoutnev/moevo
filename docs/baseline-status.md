# Dataset inventory and Astra validation

Snapshot: 2026-09-15. The inventory contains **16 benchmark candidates across 12 domains**.
The shared mini-run policy uses `gpt-6-astra`, `xhigh`, and signed-in ChatGPT-account
Codex CLI authentication. API-key fallback is disabled for its solver and judge calls.

## Current inventory

| Domain | Benchmarks |
| --- | --- |
| Finance | FinQA; BizFinBench.v2 |
| Science | GeneBench Pro public; Terminal-Bench Science |
| Mathematics | AMO-Bench P subset; PutnamBench |
| Planning | TravelPlanner |
| Operations research | ORAgentBench |
| Data analysis | DSBench |
| Professional work | GDPval; AutomationBench public |
| Software / terminal | Terminal-Bench 2 |
| Legal | Harvey LAB |
| Medicine | HealthBench Professional |
| Customer service | τ³-bench |
| Education | EduAgentBench |

GDPval and standard Terminal-Bench are retained. Data or source is downloaded for
all entries; runtime and grader validation are reported separately.

## Real mini-run results

See the [real mini-run report](mini-run-status.md) for one real task per available
benchmark, native scores, positive/negative controls, and setup blockers. See the
[shared policy and commands](../submissions/superharness-astra-v1/README.md) for
reproduction and the account-only transport. These are actual model attempts,
separate from the earlier [dummy submission](../submissions/superharness-dummy/README.md).

Some protocols differ from the official leaderboard, especially account-Astra
model judges, local Docker resource limits, and GDPval's public rubric evaluation.
A passing integration does not establish a trustworthy full-benchmark estimate.

## Earlier exploratory screens

- FinQA: 25/32 accepted answers (78.125%) in the earlier development screen.
  Incorrect or ambiguous labels were found; failures still need semantic audit.
  [Local evidence](../results/domain_validation/finqa_v2/report.json).
- The larger HealthBench Professional screen was stopped when the user narrowed
  the task to a one-case-per-benchmark integration run. It is not a completed
  baseline. The new shared-policy medical mini result is reported separately.
- TravelPlanner's initially sampled released annotation failed its own reference
  constraints. The mini run selects a released annotation that passes the
  reference controls before making a model call and records the rejected checks.
  This is a validation fixture, not representative sampling.
- AutomationBench's generated task IDs disagree with fixed IDs in the selected
  task's assertions. Its native score is preserved and explicitly flagged.

**No benchmark is yet admitted as having verified practical headroom for
multi-domain evolution.** A single passing task does not demonstrate saturation;
a defective label or broken grader does not demonstrate model weakness. Final
search/selection/test manifests are not frozen, and no evolution was run here.

Raw benchmark data, model transcripts, artifacts, and rubric details remain local
under ignored `data/` and `results/` directories. Source revisions and runtime
hashes are included in the published metadata-only mini report.

See [benchmark qualification](benchmark-expansion.md),
[additional benchmark research](additional-benchmarks.md), and
[many-objective search methods](many-objective-search.md) for the broader plan.
