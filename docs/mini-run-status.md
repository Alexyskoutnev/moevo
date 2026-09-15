# Real Astra mini-run

**14/16 benchmark integrations validated; 15 real task attempts scored.**

One task per benchmark with an available runtime. These are integration results, not full-benchmark accuracy or evidence of saturation.

Model: `gpt-6-astra` / `xhigh`. Signed-in Codex ChatGPT account; API-key fallback disabled. No evolution was run.

| Domain | Benchmark | Status | One-task native score |
| --- | --- | --- | ---: |
| Finance | FinQA | scored | 1 |
| Finance | BizFinBench.v2 | scored | 1 |
| Science | GeneBench Pro public | scored | 0.9564 |
| Science | Terminal-Bench Science | scored | 0 |
| Mathematics | AMO-Bench P subset | scored | 1 |
| Mathematics | PutnamBench | scored | 1 |
| Planning | TravelPlanner | scored | 1 |
| Operations research | ORAgentBench | scored | 1 |
| Data analysis | DSBench | scored | 1 |
| Professional work | GDPval | scored | 0.6667 |
| Professional work | AutomationBench public | needs_audit | 0.6 |
| Software / terminal | Terminal-Bench 2 | scored | 1 |
| Legal | Harvey LAB | scored | 0 (54/59 criteria) |
| Medicine | HealthBench Professional | scored | 1 |
| Customer service | tau3-bench | scored | 0 |
| Education | EduAgentBench | blocked | — |

## Interpretation

`scored` means the model attempted a real task and the evaluation path passed its controls. A score of zero can still validate an integration. `needs_audit` retains a real score with a known protocol/data issue. `blocked` has no model score; setup or evaluator failure is never counted as model failure.

The same shared instructions were used throughout. Task-specific tools and graders differ by benchmark. Positive/negative controls verify the evaluation path; they do not establish expert-level judge reliability.

## Protocol qualifications

- **FinQA:** Official reference grader; one development task; shared account-Astra policy
- **BizFinBench.v2:** Official BizFinBench.v2 numeric grader; English subset
- **GeneBench Pro public:** Public 10-case release; official composite grader
- **Terminal-Bench Science:** Official science environment, oracle and separate verifier; 1200s/2CPU/4GB offline mini-run variant; source commit pinned. The attempt reached its 80-call budget while reading image base64 through the text-only terminal and produced no required result files. No image-display tool was enabled; this is an explicit limitation of this harness configuration.
- **AMO-Bench P subset:** Official reference grader; one development task; shared account-Astra policy
- **PutnamBench:** Pinned PutnamBench theorem, Lean 4.27.0 / Mathlib kernel; restricted proof-term protocol and local budget variant
- **TravelPlanner:** Official sole-planning constraints; fixture selected for valid annotation; not a representative score
- **ORAgentBench:** Official oracle and grader; local 2CPU/4GB budget variant
- **DSBench:** Official data and judge prompt; account-Astra judge variant
- **GDPval:** Public GDPval weighted rubric and actual artifact extraction; account-Astra judge variant; not expert pairwise win rate
- **AutomationBench public:** Real task and native grader ran; no-op/synthetic-positive controls passed. Fixed assertion IDs disagree with generated tool IDs, so task validity remains under audit.
- **Terminal-Bench 2:** Official Terminal-Bench 2 test code/Python 3.13; dependencies preinstalled; local agent runtime and resource variant
- **Harvey LAB:** Official LAB criterion prompt and all-pass rule; single account-Astra judge variant; DOCX extracted with pandoc
- **HealthBench Professional:** Public HealthBench scorer; account-Astra solver/judge; text only; external protocol variant
- **tau3-bench:** Native text simulator and reward; account Astra agent/user; JSON tool transport; 50-step cap
- **EduAgentBench:** EduAgentBench: all 22 asset checksums pass, but the downloaded public release has no evaluator or Canvas runtime; cannot claim official E2E validation

## Reproduction and evidence

See the [run manifest and commands](../submissions/superharness-astra-v1/README.md) and [metadata-only report](../submissions/superharness-astra-v1/validation.json). The JSON records task identifiers, native metrics, source revisions, policy hash, and available runtime hashes. Raw model transcripts, submitted artifacts, grader rubrics, and downloaded benchmark data remain local under `results/` and `data/`.

The larger medical screen was stopped when the task was narrowed to this mini run. None of these single-task results admits a benchmark as having demonstrated useful headroom. A separate, predeclared multi-task screen is still required before harness evolution.
