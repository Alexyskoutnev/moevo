"""Mutation prompt builder for moevo.

Mirrors SkyDiscover's context_builder templates for 1:1 prompt parity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.types import Program

# ── System message (matches SkyDiscover config.py) ─────────────────────────

SYSTEM_MESSAGE = """\
You are evolving the SOURCE CODE of an AI coding agent. The agent is a single
Python file that receives a task, works in an isolated directory, and produces
output files. It is scored on task completion quality.

The entire file is yours to change — config, prompts, tools, the agent loop,
helper functions, error handling. Add or remove tools. Change the architecture.
The only constraint is the score: working code that scores higher survives.
"""


# ── Explore/exploit labels (matches SkyDiscover's AdaEvolveDatabase) ───────

EXPLORE_LABEL = """\
## PARENT SELECTION CONTEXT
This parent was selected through diversity-driven sampling to explore different regions.

### EXPLORATION GUIDANCE
- Consider alternative algorithmic approaches
- Don't be constrained by the parent's approach
- Look for fundamentally different algorithms or novel techniques
- Balance creativity with correctness

Your goal: Discover new approaches that might outperform current solutions."""

EXPLOIT_LABEL = """\
## PARENT SELECTION CONTEXT
This parent was selected from the archive of top-performing programs.

### OPTIMIZATION GUIDANCE
- This solution works well, but meaningful improvements are still possible
- You may refine the existing approach OR introduce better algorithms
- Consider: algorithmic improvements, better data structures, efficient libraries
- Ensure correctness is maintained

Your goal: Improve upon this solution."""


# ── User message templates (matches SkyDiscover's diff/full_rewrite) ───────

DIFF_TASK = """\
# Task
Suggest improvements to the program that will improve its scores on the objectives.
The system maintains diversity across these dimensions: score, complexity.
Different solutions with similar scores but different features are valuable.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

**CRITICAL**: You can suggest multiple changes. Each SEARCH section must EXACTLY match \
code in "# Current Solution" - copy it character-for-character, preserving all whitespace \
and indentation. Do NOT paraphrase or reformat.
Be thoughtful about your changes and explain your reasoning thoroughly.
Include a concise docstring at the start of functions describing the exact approach taken."""

FULL_REWRITE_TASK = """\
# Task
Suggest improvements to the program that will improve its scores on the objectives.
The system maintains diversity across these dimensions: score, complexity.
Different solutions with similar scores but different features are valuable.

Provide the complete new program solution.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```python
# Your rewritten program here
```

**CRITICAL**: Be thoughtful about your changes and explain your reasoning thoroughly.
Include a concise docstring at the start of functions describing the exact approach taken."""


def build_prompt(
    parent: Program,
    context_programs: list[Program],
    objectives: list[str],
    diff_mode: bool = True,
    explore: bool = False,
) -> tuple[str, str]:
    """Build (system, user) prompt for LLM mutation.

    Mirrors SkyDiscover's context_builder prompt assembly:
    1. Metrics & improvement areas
    2. Context programs (with scores and code)
    3. Current program (with explore/exploit label, metrics, code, feedback)
    4. Task instructions (diff or full rewrite)

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    parts: list[str] = []

    # ── Metrics & objectives ───────────────────────────────────────────
    parent_scores = ", ".join(f"{o}={parent.get_objective(o):.4f}" for o in objectives)
    parts.append("# Current Solution Information")
    parts.append(f"- Objectives (all maximized): {', '.join(objectives)}")
    parts.append(f"- Main Metrics: {parent_scores}")

    # Improvement areas: objectives below 1.0
    weak = [o for o in objectives if parent.get_objective(o) < 0.9]
    if weak:
        parts.append(f"- Focus areas: {', '.join(weak)}")
    parts.append("")

    # ── Context programs (other Pareto frontier members) ───────────────
    if context_programs:
        parts.append("# Other High-Scoring Programs (for reference)")
        for i, ctx in enumerate(context_programs):
            ctx_scores = ", ".join(f"{o}={ctx.get_objective(o):.4f}" for o in objectives)
            parts.append(f"## Program {i + 1} (scores: {ctx_scores})")
            parts.append(f"```python\n{ctx.solution}\n```")
            parts.append("")

    # ── Current program with explore/exploit label ─────────────────────
    label = EXPLORE_LABEL if explore else EXPLOIT_LABEL
    parts.append(label)
    parts.append("")
    parts.append(f"# Current Solution (scores: {parent_scores})")
    parts.append(f"```python\n{parent.solution}\n```")
    parts.append("")

    # ── Evaluation feedback ────────────────────────────────────────────
    if parent.feedback:
        parts.append("# Evaluator Feedback")
        parts.append(parent.feedback)
        parts.append("")

    # ── Task instructions (diff or full rewrite) ───────────────────────
    parts.append(DIFF_TASK if diff_mode else FULL_REWRITE_TASK)

    return SYSTEM_MESSAGE, "\n".join(parts)
