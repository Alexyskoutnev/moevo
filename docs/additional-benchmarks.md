# Additional benchmark candidates

Research snapshot: 2026-09-15. These are candidates, not an E2E-validated or
headroom-qualified suite. Keep the headline experiment near ten benchmarks;
use additional candidates to replace weak axes or as untouched transfer tests.

## Priority additions

| Domain | Benchmark | Reason to consider it | Grading and integration limits |
| --- | --- | --- | --- |
| Legal | [Harvey LAB](https://github.com/harveyai/harvey-labs) | Real legal assignments with matter documents, drafting/review/research tasks, and expert rubrics | Version the task release; all-pass grading uses LLM judges, requiring an account-CLI adapter and judge controls |
| Medicine | [HealthBench Professional](https://huggingface.co/datasets/openai/healthbench-professional) | Clinician consultation, documentation and research; physician-written rubrics | Public data, but official reported results use an internal implementation. An external account-Astra judge is a protocol variant |
| Customer service | [tau-three-bench, text mode](https://github.com/sierra-research/tau2-bench) | Interactive airline, retail, telecom and banking-knowledge tasks with policies and tools | Use the corrected release at least 1.0.1; both solver and simulated user need account transport. Banking-knowledge scores changed with grading fixes |
| Education | [EduAgentBench](https://huggingface.co/datasets/eduagentbench/eduagentbench) | Pedagogical judgment, situated tutoring and teaching workflows; a different objective from simply answering mathematics questions | Public artifact describes 150 tasks, synthetic/mock course state and non-commercial research evaluation. Full environment and grader availability still need validation |

The proposed evolvable mechanisms are document retrieval and citation checks for
law; evidence review, uncertainty and response checking for medicine; dialogue
state and action planning for customer service; and feedback sequencing and course
workflow execution for education. These are research hypotheses, not measured gains.

### Published Astra evidence for medicine

OpenAI reports **63.4 on the length-adjusted HealthBench Professional score**
(69.5 unadjusted), and explicitly prefers it to older HealthBench variants for
measuring frontier progress. This is a credible lead for headroom, not our local
baseline or a pass rate. The official dataset card says that no official external
implementation is released. Preserve raw and length-adjusted rubric metrics, and
do not present an account-Astra-judged run as reproducing the internal protocol.

Sources: [Astra system card](https://deploymentsafety.openai.com/gpt-6-astra),
[HealthBench Professional data card](https://huggingface.co/datasets/openai/healthbench-professional).

## Additional domain coverage

| Domain | Benchmark | Why it is distinct | Main limitation |
| --- | --- | --- | --- |
| Mechanical engineering / CAD | [BenchCAD CodeEdit](https://github.com/BenchCAD/BenchCAD-main) | 748 instruction-guided edits to parametric mechanical parts; execution-grounded geometry scoring without an LLM judge | Actual Astra CodeEdit headroom unmeasured; the launch-page 95.9 geometric-overlap score concerns CAD reconstruction, not all BenchCAD tracks |
| Music / notation | [OSSQ-OMR](https://github.com/MALerLab/string-quartet-omr-benchmark) | Transcribe scanned or rendered quartet scores into symbolic music; OMR normalized edit distance | Requires image access, music-format tooling and score-level splits. Do not transfer Astra's OpenScore launch score to a different release or protocol |
| Clinical / biomedical workflows | [HealthAgentBench](https://github.com/microsoft/HealthAgentBench) | 54 terminal tasks covering trial matching, EHR auditing/modeling and medical imaging | Several categories require gated data and credentials; about 30 GB of assets. Some graders use model APIs and require adaptation. Keep public executable categories labeled as subsets |

BenchCAD's four tracks have different metrics and inputs. CodeEdit is interesting
for a harness that executes, inspects and repairs geometry. The released 1.0
benchmark is available; the 2.0 agentic benchmark is still a development roadmap
according to the authors' organization page. Do not claim 2.0 is validated.

The music benchmark includes official score-level splits to prevent a movement's
related images from crossing train/test boundaries. Its CC0 data and deterministic
OMR-NED metric make it a useful potential transfer test for multimodal tool use.

## Lower-priority or evaluation-only alternatives

- [BrowseComp](https://openai.com/index/browsecomp/): useful for web research, but
  Astra's launch page reports 91.5 accuracy. Our practical-headroom screen may
  reject it. Browsing also needs a controlled search protocol and answer-leakage
  checks. Do not count search or transport failures as task reasoning failures.
- BenchCAD reconstruction: published Astra geometric overlap is 95.9, so it is a
  lower-priority core axis than a benchmark with a larger remaining gap. This
  metric is not a strict task-success percentage.
- [APEX-Agents](https://huggingface.co/datasets/mercor/apex-agents): attractive
  independent professional-work evaluation, but its card explicitly restricts
  use to evaluation and forbids training, fine-tuning and parameter fitting.
  Reserve it for a frozen final harness rather than using its scores to select
  evolutionary mutations. APEX-Agents 1.1 changed grading; pin a release.
- LegalBench / older medical exam questions: useful controls, but verify actual
  Astra headroom before making them headline evolutionary objectives.

Additional primary sources: [Astra launch results](https://openai.com/index/gpt-6-astra/),
[BenchCAD release status](https://github.com/BenchCAD-org),
[APEX-Agents 1.1 announcement](https://www.mercor.com/blog/introducing-apex-agents-1-1/).

## Immediate scope

The user explicitly asked to include GDPval and standard Terminal-Bench, and to
cover law and medicine. Preserve GDPval and download standard Terminal-Bench 2
separately from Terminal-Bench Science. Prioritize Harvey LAB and HealthBench
Professional for the two requested new domains. The other entries above are a
researched selection pool, not a claim that their assets or adapters are installed.
