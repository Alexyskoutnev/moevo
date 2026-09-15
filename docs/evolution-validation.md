# Fresh evolution validation

The current priority is verifying the new evolution loop with real account-backed
calls. The manuscript is unchanged. Historical-result audits are paused.

## Current test

- One shared instruction component; Astra generates changes and solves tasks.
- Terra grades rubric tasks; deterministic native graders keep their own rules.
- Thirteen existing diagnostic tasks across eleven domains.
- One fresh complete seed, then four mutation steps; at most 65 task attempts.
- Identical task IDs, grading protocol and resource limits within a comparison.
- Paired screens determine which candidates get the complete fixed panel.
- Only complete vectors enter population selection; failures are recorded as
  errors rather than invented zero scores. Actual scored zeros remain valid.

Run the artifact check after the epoch:

```sh
.venv/bin/python -m experiments.verify_pilot_epoch \
  --run results/pilot-epoch/astra-terra-01 \
  --output results/pilot-epoch/astra-terra-01/validation.json --require-complete
```

The check reconstructs every population vector from task records, reconciles
ledger scores with raw grader reports, checks published task IDs, requires a
changed instruction string and a fully scored child in the population history,
and verifies the selected source has a complete evaluation. A score gain is
not required to pass this plumbing check. An epoch with only screened-out children
does not demonstrate the complete confirmation/selection path.

With four mutations and twelve population slots per island, this short epoch
does not fill an island enough to exercise NSGA-III pruning. Offline selection
tests cover that path. Development feedback includes task grading details and
may contain reference values; these fixtures cannot establish generalization.
Codex account calls pin model and reasoning effort, but the generic controller's
temperature and maximum-output-token fields are not enforced by this transport.

## What makes the next measurements convincing

| Requirement | Before reporting improvement |
| --- | --- |
| Fair baseline | Compare the evolved harness with its own unevolved seed on the same tasks, backbone, judge, tools and limits. |
| Enough evidence | Expand fixed panels before estimating performance; one task is not a benchmark accuracy estimate. Check baseline headroom on development data. |
| Genuine evolution | Retain changes and regressions. Report raw candidate trajectories and distinguish them from the selected incumbent. |
| Independent searches | Repeat whole searches with independent seeds; repeated scoring of one harness measures a different source of variation. |
| Matched controls | Give each selector the same task access, starting harness and resource budget. Keep commercial-harness comparisons separate. |
| Final evaluation | Freeze selection before evaluating untouched tasks. Never send final-test feedback to the proposer. |
| Safety | Test that the objective responds to actual harness changes, examine case-level failures and over-refusal, and keep a benchmark score distinct from deployment safety. |
| Resources | Count failed attempts as well as successful calls, record model usage and wall time, and label unavailable account dollar costs as unknown. |

The diagnostic dashboard uses mean normalized percentages with explicit task
counts. The overall score gives equal weight to each domain; binary accuracy,
rubric scores and solution-quality rewards retain distinct metric labels. Export
tables and figures using:

```sh
.venv/bin/python -m experiments.evolution_dashboard \
  --run results/pilot-epoch/astra-terra-01 \
  --export-results results/pilot-epoch/astra-terra-01/export
```

The export is a record of this diagnostic run, not publication evidence of
generalization. New `moevo/revision/` code remains separate experimental scaffolding;
it is not used by the active pilot.
