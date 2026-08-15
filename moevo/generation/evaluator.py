"""Evaluator bridge - loads user's evaluate() and runs it in a thread."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import tempfile
from pathlib import Path

from ..core.types import EvalResult

logger = logging.getLogger("moevo.evaluator")


def load_evaluate_fn(evaluator_path: str):
    """Load the evaluate() function from a user-provided module."""
    path = Path(evaluator_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Evaluator not found: {path}")

    spec = importlib.util.spec_from_file_location("moevo_user_eval", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluator from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["moevo_user_eval"] = module
    spec.loader.exec_module(module)

    fn = getattr(module, "evaluate", None)
    if fn is None:
        raise AttributeError(f"Evaluator module {path} has no evaluate() function")
    return fn


async def run_evaluation(evaluate_fn, code: str, program_id: str) -> EvalResult:
    """Write code to a temp file and call evaluate(path) in a thread."""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix=f"moevo_{program_id}_",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = await asyncio.to_thread(evaluate_fn, temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

        if isinstance(result, dict):
            metrics = {
                k: v for k, v in result.items() if isinstance(v, (int, float)) and k != "artifacts"
            }
            artifacts = result.get("artifacts", {})
            if not isinstance(artifacts, dict):
                artifacts = {}
            return EvalResult(metrics=metrics, artifacts=artifacts)

        # Support EvaluationResult-like objects with .metrics attribute
        if hasattr(result, "metrics"):
            metrics = dict(result.metrics) if result.metrics else {}
            artifacts = (
                dict(result.artifacts) if hasattr(result, "artifacts") and result.artifacts else {}
            )
            return EvalResult(metrics=metrics, artifacts=artifacts)

        logger.warning("Unexpected evaluate() return type: %s", type(result))
        return EvalResult(error=f"Unexpected return type: {type(result)}")

    except Exception as e:
        logger.error("Evaluation failed for %s: %s", program_id, e, exc_info=True)
        return EvalResult(error=str(e))
