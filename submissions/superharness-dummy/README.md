# Dummy SuperHarness submission

One unevolved, shared policy around **gpt-6-astra / xhigh**, using the signed-in
Codex ChatGPT account. The manifest includes all 16 candidate benchmarks.

This local submission runner initially connects **FinQA and AMO-Bench**. The
remaining candidates appear as `not_run` with null scores. Existing benchmark
pilots remain separate until their adapters are connected to this shared entry.

From the repository root:

```sh
# Check packaging and both actual reference graders with deliberately empty answers.
# Makes no model calls. Requires downloaded data and the existing Docker image.
.venv/bin/python experiments/run_submission.py --mode dummy --output results/submissions/my-dummy-run

# Run one fresh development task per connected benchmark through the SAME policy.
.venv/bin/python experiments/run_submission.py --mode smoke --output results/submissions/my-astra-smoke
```

Each run writes `submission.zip`, `submission.json`, `predictions.jsonl`,
`summary.json`, checksums, and task-local grader/agent traces. The ZIP contains
only the policy and public result records; task data, gold answers and account
credentials are excluded. Execute the policy using this Moevo checkout and its
separately downloaded benchmark assets. Nothing is uploaded to a leaderboard.

Dummy scores are plumbing checks. A live one-task score is an integration smoke
test, not a full benchmark result, SOTA claim, or an estimate of headroom. Both
modes are marked ineligible for evolution. Native metrics are retained, and
infrastructure failures have null scores. Existing output directories are never
overwritten.
