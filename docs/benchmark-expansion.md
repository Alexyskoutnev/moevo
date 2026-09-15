# A multi-domain SuperHarness benchmark suite

Working audit, 2026-09-15. This supersedes the earlier software-focused proposal.
**No seven-domain suite is yet qualified end to end with measured Astra headroom.**
Downloaded sources are not equivalent to ready benchmarks.

Scope expanded at the user's request: retain GDPval, include standard
Terminal-Bench alongside Terminal-Bench Science, and add law and medicine.
[Additional benchmark research](additional-benchmarks.md) prioritizes Harvey LAB
and HealthBench Professional and records optional distinct-domain candidates.

## Objective

Evolve one complete harness around frozen `gpt-6-astra`, using the signed-in
Codex CLI account for solver and proposer calls. Evaluate the same artifact on
finance, science, mathematics, planning, operations research, data analysis, and
professional work. Aim to improve all domains; saturation and SOTA require actual
measurement on an untouched final set.

A shared loop can include reusable specialist skills, but routing and all
components must be frozen before final evaluation. Do not assemble the headline
radar curve from a different best harness per benchmark. Algorithm choice and
the Sener–Koltun paper are covered in [the search audit](many-objective-search.md).

## Candidate inventory

| Domain | Candidate | Local evidence | Remaining qualification |
| --- | --- | --- | --- |
| Finance | [BizFinBench.v2](https://github.com/HiThink-Research/BizFinBench.v2), offline English tasks | Source and data downloaded; Astra passed 1/1 numeric task with official grading | Representative screen across finance categories and semantic audit |
| Science | [Terminal-Bench Science 0.1](https://github.com/harbor-framework/terminal-bench-science) | Task source downloaded | Pin release rather than main; environments, official verifier controls, and Astra runs |
| Mathematics | [PutnamBench](https://github.com/trishullab/PutnamBench); [AMO-Bench](https://github.com/meituan-longcat/AMO-Bench) as an informal alternative | Putnam source downloaded; AMO data downloaded, Astra passed 1/1 using official parser-only grading | Putnam Lean/Mathlib setup; AMO representative screen (39 parser-graded tasks, too small for the current core admission procedure) |
| Planning | [TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner) | Source, tasks, reference information and full offline database downloaded; official evaluator executes | Sampled train annotation fails its own city-consistency constraint; audit positive control before Astra run |
| Operations research | [ORAgentBench](https://github.com/ORAgentBench/ORAgentBench) | Astra passed feasibility and scored quality 2/2 on one task; oracle and negative controls passed | Harder representative screen and solver replay; smoke budget differs from official task budget |
| Data analysis | [DSBench](https://github.com/LiqiangJing/DSBench) | All analysis assets downloaded; Astra passed 1/1 question with tools and validated judge controls | Representative screen; account-Astra replacement of original GPT-4o judge is a labeled protocol variant |
| Professional work | [AutomationBench public](https://github.com/zapier/AutomationBench) | Live Astra run made 26 simulated tool calls, earned 0.6 partial credit, strict completion 0/1 | Audit conflicting tool IDs before counting the result as headroom; representative baseline |

This is a candidate list, not the promised ready list. Revisions and archive
hashes are in `data/external/*/source_manifest.json`; Hugging Face file hashes are
in `data/raw/*/_moevo_manifest.json`. Downloads are ignored by git.

Terminal or Python use does not make science, optimization, and data analysis
software engineering. Conversely, multiple similar code benchmarks do not establish
independent domains. Check how domain scores respond to actual harness changes.

The [Astra launch page](https://openai.com/index/gpt-6-astra/) identifies headroom
on AutomationBench, Terminal-Bench Science, and HLE with tools, while FrontierMath
Tier 4 is close to saturation. These are leads, not our measurements: public data,
model configuration, and inference protocol can differ. AutomationBench public is
not its private leaderboard.

### Access and evaluation limitations

- HLE and DataSciBench need dataset access unavailable here. DataSciBench returned
  an authorization error. Neither is ready; do not substitute an unofficial copy.
- DABstep's tasks and seven context files downloaded. Its 450 main tasks have blank
  public answers; only 10 development tasks expose answers. This supports a small
  local smoke test, not a locally scored 450-task evolution objective. Official
  submission can serve as an external test.
- GeneBench Pro public downloaded with inputs and graders but has only 10 released
  cases. It is a small transfer set, not the full launch benchmark.
- SpreadsheetBench 2 needs additional assets/runtime and an explicit account-only
  protocol for any model-judged visualization tasks.
- Older mathematics benchmarks may be saturated or contain statement defects;
  downloadable data alone is insufficient.

## Actual integration evidence

All measurements below use an **unevolved Astra harness with tools**, not a
tool-free model. Model and effort are `gpt-6-astra` / `xhigh` through the signed-in
Codex account. These are local development checks with different task budgets,
not published full-benchmark baseline scores. Current evidence is summarized in
[the baseline status table](baseline-status.md).

### FinQA

The corrected pilot at `results/domain_validation/finqa_v2` completed 32 real
Astra xhigh account-CLI runs: **25/32 were accepted by the official program
interpreter**. This is a seeded training-set screen, not a held-out result.

Gold programs reproduced their stored answers and invalid-program controls failed.
However, semantic review found an incorrect gold program for a three-year average,
plus ambiguous sign, percentage, and table-selection cases. The seven rejected
answers are not seven verified reasoning failures. The 78.125% acceptance rate
therefore does not establish practical headroom. Prefer BizFinBench.v2 pending
validation; retain FinQA as an integration fixture until its labels are audited.

An earlier prompt introduced percentage-unit ambiguity. Its logs remain in
`results/domain_validation/finqa`; do not combine them with the corrected screen
or count errors in our prompt as model reasoning failures.

### AutomationBench

The adapter loads the official synthetic world, tools, service permissions, and
state assertions. It does not access real email, Slack, financial services, or an
inference SDK. MCP connects only to a task-local synthetic world.

The first account run reached Astra, but missing tool approval configuration
prevented every action. Its zero score is an infrastructure failure. The corrected
adapter authorizes only its two known simulated tools for that invocation and
rejects E2E validation unless an action actually executes.

The corrected run completed 26 tool calls and scored 0.6 partial credit, with
strict completion 0/1. The synthetic task exposes inconsistent Asana task IDs;
the two failed assertions require a fixed ID while the action response also
supplies a different generated ID. Until audited, this is not a verified model
reasoning failure.

### New domain checks

- BizFinBench.v2 numeric computation: **1/1**, official grader, positive and
  negative controls passed. Other downloaded finance categories have not yet
  been baseline-scored.
- GeneBench Pro public: **1/1 passed**, native composite score **0.9989** on
  `wf_selection`. The public release contains only 10 cases.
- ORAgentBench: **1/1 feasible**, native quality **2/2**, normalized scalar reward
  **1.0** on `industrial_water_reuse_blending`. The local smoke budget is 2 CPU,
  4 GiB, 1200 seconds; it is not the original task's resource protocol.
- AMO-Bench: **1/1** on seeded P-subset index 36, using the original parser
  functions and pinned dependencies in a separate network-disabled container.
  The 11 description-graded tasks are excluded, preventing API judge calls.
- TravelPlanner: the official evaluator runs, but train annotation 41 places
  San Angelo and Houston attractions on days spent in San Antonio. Its own
  official evaluator rejects the annotation. No Astra score is reported yet.
- DSBench: assets include all 466 analysis questions across 38 competitions.
  Astra passed **1/1**, competition `00000034`, question 9; judge positive and
  negative controls passed.
  Its published scorer uses a GPT-4o API judge. The local adapter preserves the
  original judge prompt and substitutes account-based Astra; results must be
  labeled as a protocol variant, not original-judge leaderboard results.

## Admission procedure

1. Pin data, evaluator, environment, model, effort, and resource limits. Inventory
   all required assets and check access and licenses.
2. Verify scoring with known-correct and deliberately wrong solutions. Agreement
   between a gold program and its stored label is not a semantic correctness check.
3. Give Astra only instructions and allowed inputs. Keep answers, grading code,
   hidden tests, and other model submissions outside the agent workspace.
4. Run task → Astra → tools/environment → official grader. Record traces, usage,
   completion, runtime, and infrastructure errors separately.
5. Screen a predeclared representative panel. The current helper requires at least
   32 tasks, success at most 90%, and at least 10% independently verified failures
   (minimum two) to confirm headroom on that panel. These are experiment thresholds,
   not full-benchmark claims. Smaller public sets stay labeled transfer checks.
6. Verify that plausible harness changes affect results under equal budgets.
7. Freeze separate search, selection, and final-test manifests. Group related
   questions by report, environment, or task family before splitting.
8. Then publish the ready list and launch the multi-domain search comparison.

Preserve native metrics alongside optimization objectives. Define success for
partial-credit metrics before examining results; partial credit is not completion.

## Existing pipeline findings

The generic search engine accepts more objective names, but the original
GDPval/ToolEmu experiment has two-objective assumptions in its runner, carry-forward
selection, feedback concatenation, and reporting. These need coordinated changes.

The legacy safety evaluator extracts the candidate model name and makes a separate
fixed-prompt call instead of invoking the evolved harness. A recorder showed
identical downstream arguments for opposite candidate prompts; neither `run()`
executed. This path cannot measure harness safety improvement without repair.
This is evidence about this checkout, not every historical experiment.

The empty-safety-set reward of 1.0 was replaced by an error. Other fixes include
required metric validation, constant-axis crowding, reproducible RNG/checkpoints,
and scalar selection. Historical seed files remain unchanged.

The new account client uses `codex exec` with ChatGPT login, strips API-key
environment variables, and defaults to `gpt-6-astra`. `MOEVO_ACCOUNT_ONLY=1`
rejects legacy provider routes. Existing provider code remains for reproduction
and is not used by these pilots.

## Reporting

Each joint-method radar curve must show one frozen artifact on every domain, with
consistent axes and a table of scores and uncertainty. Include raw Astra, the fixed
seed harness, scalar evolution, NSGA-II, and the selected alternative. Clearly
label separately evolved specialist references.

Report the weakest domain, mean, regressions, resource usage, and progress versus
evaluation budget. A finite-sample 100% only describes that test set. SOTA requires
matching benchmark versions and comparable inference and task budgets.
