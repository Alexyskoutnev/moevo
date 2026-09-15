# Harness implementation review

Reviewed 2026-09-15 using current source and current-session diagnostic artifacts.
No manuscript edits or historical paper-result audit were performed.

## What is being evolved

The candidate is a Python file containing one literal `INSTRUCTIONS` string.
The adapters read this string with an AST parser; they do not execute generated
Python. Astra uses those same instructions on every benchmark. Model, tools,
task limits, inputs, and graders are outside this mutation space.

This experiment tests shared-instruction evolution. It does not yet test changes
to the executable tool loop or harness architecture. The fixed execution source
is viewable separately in the dashboard when it matches the run's recorded hash.

## Verified connections

- Every one of the 13 adapter paths applies the candidate instructions. The
  retail simulator applies them to the agent only; its customer behavior stays fixed.
- Generation and solving use account Astra/xhigh. Rubric judges use Terra/medium.
  Native program, constraint, proof, simulator, and numeric graders retain their
  distinct protocols. No inference API key is used on this path.
- Public task files reach the solver. Hidden rubrics and oracle solutions remain
  grader-side. The new sliced adapter omits answer keys from mutation feedback.
- A parent and child use identical task IDs in each paired screen. Only complete
  vectors on the fixed current panel enter the population and aggregate charts.
- The selected result is one actual candidate, chosen by its weakest domain
  first. It is not a synthetic combination of the best scores from different agents.
- A failed answer remains in the score denominator. Two discovered bugs were
  corrected before starting `astra-terra-new-02`: missing OR submissions and
  corrupt submitted documents now receive explicit zero scores; genuine runtime
  failures remain separate errors.

## Actual mutation inspected

The first mutation in the current-session thirteen-fixture diagnostic expanded
the 246-character seed instructions to 4,925 characters: requirements, evidence,
dependencies, quantitative checks, legal analysis, authorized actions, artifact
structure, and final verification. It passed the literal-only constraint. No
hardcoded task answers or IDs were apparent on review.

Its first paired screen produced zero change on data analysis, optimization, and
legal; it was screened out. This is an observed unchanged result, not a gain.
The fresh-task run has its own source, task panel, and results. The two dashboards
keep them separate.

## Fixes prepared for the eight-slice launcher

The running one-slice bridge retains its original generation protocol. The
not-yet-launched eight-slice runner now has:

- A dedicated instruction-only prompt, removing conflicting architecture-change guidance.
- Atomic SEARCH/REPLACE validation: every block must match exactly once, and
  malformed or partly applicable patches are rejected.
- Literal scope and changed-instruction checks before candidate task evaluation.
- Durable request, parent source/hash, parsed candidate, unified diff, status,
  actual account events, and model-usage artifacts for each proposal.
- Explicit resource accounting: 32 proposal calls and up to 728 task attempts.

For the current bridge, a separate observer preserves temporary candidate source
only after its hash matches a real task ledger entry. It changes no evaluator or
scores. It cannot recover source that disappeared before observation. The
dashboard distinguishes in-progress, screened-out, and fully tested candidates.

## Limits that still matter

- Four proposals with twelve slots per island do not exercise NSGA-III pruning.
  This is pipeline validation, not evidence that NSGA-III beats another selector.
- A zero-valued objective collapses the current zero-reference hypervolume;
  gains on other domains can be invisible to that island-reward signal.
- The bridge's generic account generation path does not enforce the displayed
  temperature/output-token parameters or preserve all proposal usage. Astra/xhigh
  is enforced. The future launcher records the actual account settings and usage.
- Resume preserves completed checkpoints and charged attempts. Interrupted pending
  proposals are archived but are not replayed exactly; a new proposal is charged.
- Auxiliary document/Lean/terminal image tags and shared retail data/runtime need
  complete immutable identities before a long resumed or publication experiment.
- One task per benchmark cannot establish accuracy, uncertainty, generalization,
  or safety. Independent full searches, matched baselines, larger development
  panels, and a fresh frozen final evaluation remain separate work.

## Views and commands

```sh
python -m experiments.evolution_dashboard --run results/first-slice/astra-terra-new-02 --port 8765
python -m experiments.capture_evolution_sources --run results/first-slice/astra-terra-new-02
python -m experiments.watch_evolution_run --run results/first-slice/astra-terra-new-02 --wandb offline
```

The **Harness code & changes** section shows the exact candidate source, a diff
against its seed, readable instructions, source hash, and evaluation status.
W&B tracking is local/offline; completion triggers exports and the real-epoch verifier.
