# Security

## Read this before running anything

**MOEvo executes code written by a language model, and it does not sandbox it.**
That is inherent to what the project does — it evolves an agent harness by
rewriting its source and running it. But it means running this repo on a machine
you care about is not safe. Specifically:

1. **LLM-authored Python is imported and executed in-process.** Candidate
   programs are written to disk and loaded with `importlib`, in the same
   interpreter as the evolution loop. There is no seccomp filter, no container,
   no separate user.

2. **The agent under evolution runs shell commands with `shell=True`, inheriting
   your full environment — including your API keys.** An evolved harness that
   decides to `echo $OPENAI_API_KEY` somewhere can do so. The keys are in the
   environment because the agent legitimately needs them.

3. **The per-task "workspace" is a working directory, not a sandbox.** Tool calls
   are path-checked against it (`moevo/harness/tools.py::_check_path`), but the
   `bash` tool beside them is not path-restricted at all. Treat the check as
   protection against mistakes, not against a determined agent.

4. **The commercial baselines are launched with their approval prompts disabled**
   — `--dangerously-skip-permissions` (Claude Code), `--yolo` (Gemini CLI),
   `--full-auto` (Codex CLI). This is necessary to run them unattended across
   hundreds of tasks, and it means they will take file and shell actions without
   asking.

**Run this inside a container or a disposable VM, with API keys scoped to a
throwaway project and a spend limit set.** Do not run it on a workstation that
holds credentials or data you would mind losing.

## Adversarial input reaches the mutation model

Evaluator feedback is interpolated into the mutation prompt. That feedback
derives from judge commentary on agent output, which is itself produced in
response to deliberately adversarial safety-benchmark prompts. There is a path,
in principle, from benchmark text to the model that rewrites the agent's source.
We do not currently escape or fence that channel.

## Reporting a vulnerability

This is research code released to accompany a paper; it is not a maintained
product and has no security-response SLA. If you find something, please open a
GitHub issue. If you would rather not disclose publicly, say so in the issue
without details and we will follow up.
