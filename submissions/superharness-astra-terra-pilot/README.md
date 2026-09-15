# Astra solver / Terra judge pilot

- Mutator and task solver: `gpt-6-astra`, extra-high reasoning.
- LLM rubric judges: `gpt-5.6-terra`, medium reasoning.
- Native deterministic graders: unchanged.
- Authentication: signed-in Codex ChatGPT account only; no inference API keys.

The [manifest](submission.json) pins the two model roles separately. Its solver
instructions match the prior baseline, so the new judge protocol is explicit.
The previous `superharness-astra-v1` manifest and results are preserved and still
use Astra judges. Scores across different judge protocols are not interchangeable.

## Real judge controls

All **10 controls passed**: DSBench 2, GDPval 2, Harvey LAB 2, HealthBench 4.
These exercise actual account Terra calls through each rubric's prompt and parser,
including positive/negative cases and medical penalty/injection handling.
No solver was called in this control run. They establish transport and rubric
polarity, not expert-level judgment or full-benchmark calibration.

Evidence: [metadata-only control report](judge-controls.json). Raw control logs
are local at `results/judge-controls/terra-medium-v2`. The earlier `v1` attempt
failed to initialize Codex inside the filesystem sandbox; it produced no valid
judge scores. The successful invocation ran with the authorized host runtime.

```sh
MOEVO_ACCOUNT_ONLY=1 .venv/bin/python -m experiments.check_judge_routing \
  --output results/judge-controls/NEW_RUN
```

The [diagnostic epoch and live dashboard](../../docs/evaluation-schedule.md)
evaluate a fresh seed and mutations under this same judge configuration.
Larger panels and judge agreement checks are still required before a benchmark
improvement claim. Nothing in this directory is an external leaderboard submission.
