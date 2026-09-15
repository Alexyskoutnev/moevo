# Choosing search for one SuperHarness

Research and design audit, 2026-09-15. No optimizer has yet won a matched
multi-domain experiment in this checkout. NSGA-III is an opt-in implementation,
not a replacement default established by results.

## Decision

Separate three jobs: proposing useful harness edits, retaining useful alternatives,
and choosing one deployable harness. Changing NSGA-II's survival operator addresses
only the second job.

Our main research hypothesis is that **trace-driven evolution with a balanced
incumbent and a diverse archive** will be more sample-efficient than generic
mutation with NSGA-II on this expensive, discrete problem. Test that hypothesis;
do not describe it as an established result or a new algorithm before comparison
with the closest existing work.

## What Sener and Koltun contributes

[Multi-Task Learning as Multi-Objective Optimization, NeurIPS 2018](https://arxiv.org/abs/1810.04650)
provides the formulation and the right experimental question: can a shared system
perform well on all tasks simultaneously? Its shared model and task-specific heads
suggest an analogy to a shared agent loop with reusable skills. That analogy does
not make skill modules differentiable or establish positive transfer.

For differentiable task losses, exact MGDA chooses nonnegative weights summing to
one to minimize the squared norm of their weighted gradient. The negative combined
gradient is a common local descent direction when it is nonzero. MGDA-UB computes
weights using representation gradients to reduce backward-pass overhead. Our
frozen Astra account interface exposes neither kind of gradient. Arbitrary prompts,
tool definitions, and Python control flow also have no direct differentiable
parameterization. Therefore neither Frank-Wolfe nor MGDA-UB is a drop-in search
replacement here.

Section 3.1 distinguishes Pareto stationarity from Pareto optimality. Neither
condition means that all losses are zero or all benchmarks are saturated.

The radar figure is applicable: each joint-method curve must correspond to one
complete, frozen artifact evaluated on every axis. Separately evolved specialists
can appear as an explicitly labeled reference; their per-axis maximum is not a
single SuperHarness.

### Keep the scalar baseline

The introduction's suggestion that weighted sums are only valid for noncompeting
tasks is too strong as a general optimization claim. With strictly positive
weights, a global weighted-sum minimizer is Pareto optimal: any dominating point
would have a strictly smaller weighted sum. Fixed weights express preferences;
they can hide unacceptable regressions, and weighted sums need not recover the
whole front in a nonconvex problem.

Later experiments in
[In Defense of the Unitary Scalarization, NeurIPS 2022](https://arxiv.org/abs/2201.04122)
found that a well-tuned sum with ordinary regularization could match or improve
specialized MTL optimizers in their settings. This does not decide our black-box
problem, but is a reason to retain a strong scalar control.

## Candidate methods and their actual roles

| Method | Useful role | Limitation for this project |
| --- | --- | --- |
| NSGA-II | Reproduction baseline; dominance and crowding survival | More objectives can leave much of a small population nondominated, weakening rank-based pressure; measure this fraction rather than assume failure |
| NSGA-III | Reference directions maintain coverage when dominance gives little discrimination | Coverage of trade-offs is not a preference for one balanced harness; small populations and noisy scores can make niches unstable |
| MOEA/D | Neighboring scalar subproblems can share useful changes; include a balanced target subproblem | Requires a decomposition and normalization; generic continuous-problem wins do not establish agent-search wins |
| Reflective evolution / GEPA | Turn failed execution traces into targeted mutations; merge complementary components | Its default per-key frontier can omit a generalist that is never best on one key; retain a balanced incumbent explicitly |
| Domain-balanced epsilon-lexicase | Preserve candidates solving different failure cases | A secondary ablation; flat sampling of unequal datasets would overweight larger domains |
| MGDA / MGDA-UB | Relevant if we later train differentiable shared parameters with gradients | Not directly applicable to account-only, frozen-model harness evolution |

[NSGA-III original paper](https://www.egr.msu.edu/~kdeb/papers/k2012010.pdf)
motivates reference-point survival for many objectives.
[MOEA/D original paper](https://doi.org/10.1109/TEVC.2007.892759)
decomposes search into scalar subproblems that exchange neighboring information.
[GEPA](https://arxiv.org/abs/2507.19457) supplies a closely related reflective
evolution baseline. Its
[candidate-selection documentation](https://gepa-ai.github.io/gepa/guides/candidate-selection/)
explicitly distinguishes its per-key frontier from the full nondominated set.
[Epsilon-lexicase analysis](https://arxiv.org/abs/1709.05394) motivates case-level
diversity; domain grouping here is our proposed adaptation.

[StarHarness](https://arxiv.org/html/2608.24804v1) is especially close prior work:
it evolves agent harness components and separates proposer-visible search tasks,
proposer-hidden selection tasks, and final evaluation. Its enterprise results do
not establish performance on our seven-domain suite. Compare against its search
protocol before claiming novelty from evolving a harness or stratifying failures.

## Proposed black-box analogue of common improvement

For a harness H, retain the full vector s(H) of domain success rates. For a proposed
edit H', evaluate paired tasks and record delta[d] = s[d](H') - s[d](H), together
with item-level failures, usage, and tool traces. This delta is an observed outcome,
not a gradient. Combining two edits is not a convex combination of their scores;
every merged harness must be evaluated again.

1. Sample failures from weak domains and inspect failed tool execution or reasoning.
   Give the proposer bounded feedback from every domain, including regressions.
2. Propose a concrete edit to instructions, tool schemas, reusable skills,
   verification, recovery, memory, or budget allocation. Do not change the model,
   the grading code, benchmark labels, or the global inference budget.
3. Screen on a balanced common panel. Missing assets, broken tool access, and grader
   exceptions invalidate the evaluation; they are not zero-valued reasoning scores.
4. Keep useful trade-off candidates in the exploration archive. Requiring strict
   improvement on every axis for every search step can prevent useful intermediate
   changes, especially with discrete and noisy scores.
5. Promote a complete candidate only after reevaluation on the selection panel.
   Use a predeclared per-domain noninferiority tolerance, paired uncertainty, and
   evidence of improvement on the weakest domain. An inconclusive comparison asks
   for more evidence; it is not automatically a promotion.
6. Freeze the final artifact and evaluate it once on untouched test tasks. No
   additional editing or choosing candidates using final-test outcomes.

### Expressing the user's target

With comparable success rates in [0, 1], minimize the largest remaining failure
rate: max_d(1 - s[d]). Equivalently, maximize min_d s[d]. Break exact ties by the
second-worst domain and continue lexicographically. This is the current optional
`balanced_champion` helper, not yet the default controller selection.

Example: scores (1, 1, 1, 1, 1, 1, 0.1) have a higher average than (0.8, ..., 0.8),
but the latter is preferable under this declared all-domain objective. A Pareto
archive can retain both. Max-min alone can still exchange one domain's performance
for another; use the separate promotion check to protect established performance.

For unlike native metrics, preserve native reporting and predeclare meaningful
success criteria or target gaps. Do not rescale using each generation's min/max:
that changes the meaning of progress. Do not count efficiency as an eighth domain;
hold resource budgets fixed for quality comparisons and report cost separately.

## Small, diagnostic experiment before a large run

First measure baseline headroom and adapter correctness in every admitted domain.
Then use the same seed harness, model/effort, tasks, mutation prompt, population
capacity, and task-evaluation budget for these search arms:

1. Scalar mean control.
2. NSGA-II survival.
3. NSGA-III survival.
4. Balanced-target decomposition with an exploration archive.

Apply identical reflective mutation to all four to isolate selection. In a second
ablation, compare generic versus trace-driven mutation under the best selection
rule. Do not confound a new selector with extra calls, a better proposer, or an
easier panel. Start with three predeclared search seeds and increase replication
if uncertainty prevents choosing a method. Account for proposer calls, solver
calls, tokens, wall time, and benchmark environment time.

Record weakest-domain success, per-domain success and confidence intervals,
strict task completion, regressions, nondominated-population fraction, and area
under the balanced-progress versus evaluation-budget curve. Hypervolume is a
secondary archive metric, with a fixed reference point; it does not select the
deployment artifact. Exact hypervolume may become costly with 7–10 objectives.

Do not reserve a large untouched holdout only to exhaust it through repeated
selection. Split by independent task family/report/environment where possible;
many questions from one source are not independent datasets.

## Implementation status

- Opt-in `selection="nsga3"`: reference-direction survival with reproducible RNG.
- `balanced_champion`: selects one real candidate, prioritizing its weakest domain.
- Tests cover survival, replay after checkpoint, and rejection of unscaled scores.
- MOEA/D, paired statistical promotion, and the multi-domain experiment above are
  not implemented or run yet. No evidence currently establishes which wins.

## Mathematical audit note on the pasted MGDA-UB proof

Independent algebra check, not a claim taken from another paper: Appendix A's
step from equation (9) to (10) does not follow merely because M is positive
definite. Positive definiteness preserves the sign of v^T M v for nonzero v,
not arbitrary cross-products u^T M v.

For example, let representation gradients be a=(1,0), b=(0,1), whose minimum-norm
convex combination is q=(0.5,0.5). Take the full-rank gradient map
J=[[1,-2],[0,1]]. Then M=J^T J=[[1,-2],[-2,5]] is positive definite, but
q^T a=0.5 while q^T M a=-0.5. Thus a shared update -Jq increases task 1 to first
order. This demonstrates why we should not transfer the stated full-rank-only
common-descent argument to a new method. It does not negate the paper's reported
empirical results or the exact MGDA minimum-norm property.
