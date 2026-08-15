"""moevo evaluator — multi-objective fitness function for code evolution.

Wraps the existing GDPval + safety evaluation pipeline to return separate
objectives (gdpval_score, safety_score) instead of a single combined_score.

Configuration via environment variables (same as evaluator.py):
    EVOLVE_SLICE          — zipper slice name (default: S1)
    EVOLVE_SAMPLE_SIZE    — GDPval tasks per evaluation (default: 3)
    EVOLVE_WORKING_DIR    — workspace directory
    EVOLVE_AGENT_MODEL    — model override for the agent
    EVOLVE_JUDGE_MODEL    — model override for the LLM judge
    EVOLVE_SAFETY_SAMPLES — safety tasks per evaluation (default: 3)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from moevo.evolve.evaluator import (
    MAX_FEEDBACK_LEN,
    _get_config,
    _load_agent_from_code,
    _run_gdpval_eval,
    _run_safety_eval,
)

logger = logging.getLogger("moevo.eval")

_eval_counter = 0


def evaluate(program_path: str) -> dict:
    """moevo entry point — returns dict with gdpval_score, safety_score, and feedback.

    Both objectives are separate (not combined), so moevo's NSGA-II can
    discover the full Pareto frontier of capability-safety tradeoffs.
    """
    global _eval_counter
    _eval_counter += 1
    eval_id = _eval_counter

    try:
        code = Path(program_path).read_text()
        if not code.strip():
            return {"gdpval_score": 0.0, "safety_score": 0.0}
    except Exception as e:
        logger.error("[eval #%d] Read failed: %s", eval_id, e)
        return {"gdpval_score": 0.0, "safety_score": 0.0}

    # Compile check
    try:
        compile(code, "<evolved_agent>", "exec")
    except SyntaxError as e:
        logger.warning("[eval #%d] SYNTAX ERROR: %s", eval_id, e)
        return {"gdpval_score": 0.0, "safety_score": 0.0}

    cfg = _get_config()
    # Force safety evaluation on
    cfg["safety_weight"] = 1.0

    # Load agent
    agent_model = cfg.get("agent_model")
    logger.info(
        "[eval #%d] Loading evolved agent (%d lines, model=%s)...",
        eval_id,
        code.count("\n") + 1,
        agent_model or "default",
    )
    agent = _load_agent_from_code(code, model=agent_model)
    if agent is None:
        logger.warning("[eval #%d] LOAD FAILED", eval_id)
        return {"gdpval_score": 0.0, "safety_score": 0.0}
    logger.info("[eval #%d] Agent loaded: %s", eval_id, agent.name())

    t0 = time.monotonic()
    try:
        # GDPval evaluation
        gdpval_score, gdpval_metrics, gdpval_feedback = _run_gdpval_eval(agent, eval_id, cfg)

        # Safety evaluation
        safety_score, safety_metrics, safety_feedback = _run_safety_eval(agent, eval_id, cfg)

        elapsed = time.monotonic() - t0
        logger.info(
            "[eval #%d] DONE — gdpval=%.1f%% safety=%.1f%% %.0fs",
            eval_id,
            gdpval_score * 100,
            safety_score * 100,
            elapsed,
        )

        # Build feedback for mutation LLM
        feedback_parts = [
            f"GDPval: {gdpval_score:.0%} ({int(gdpval_metrics.get('gdpval_completed', 0))}/{int(gdpval_metrics.get('gdpval_tasks', 0))} tasks)",
        ]
        if gdpval_feedback:
            feedback_parts.append(gdpval_feedback)
        if safety_feedback:
            feedback_parts.append(f"\nSafety: {safety_score:.0%}")
            feedback_parts.append(safety_feedback)
        feedback_str = "\n".join(feedback_parts)
        if len(feedback_str) > MAX_FEEDBACK_LEN:
            truncated = feedback_str[:MAX_FEEDBACK_LEN].rsplit("\n", 1)[0]
            feedback_str = truncated + "\n... (truncated)"

        return {
            "gdpval_score": gdpval_score,
            "safety_score": safety_score,
            **gdpval_metrics,
            **safety_metrics,
            "artifacts": {"feedback": feedback_str},
        }

    except Exception as e:
        logger.error("[eval #%d] RUNTIME ERROR: %s", eval_id, e, exc_info=True)
        return {"gdpval_score": 0.0, "safety_score": 0.0}
