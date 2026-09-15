"""Explicit account-only judge routing, independent of the Astra solver."""

from __future__ import annotations

from .client import DEFAULT_MODEL, run_codex

PILOT_JUDGE_MODEL = "gpt-5.6-terra"
PILOT_JUDGE_EFFORT = "medium"


def judge_options(policy: dict | None = None) -> dict[str, str]:
    """Old frozen policies retain Astra; new policies pin both judge fields."""
    policy = policy or {}
    keys = {"judge_model", "judge_reasoning_effort"}
    if keys.intersection(policy) and not keys.issubset(policy):
        raise ValueError("Pin both judge_model and judge_reasoning_effort")
    model = policy.get("judge_model", DEFAULT_MODEL)
    effort = policy.get("judge_reasoning_effort", "xhigh")
    if model not in {"gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}:
        raise ValueError("Judge must use a supported signed-in Codex model")
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Invalid judge reasoning effort")
    return {"model": model, "effort": effort}


def judge_metadata(policy: dict | None = None) -> dict[str, str]:
    options = judge_options(policy)
    return {"judge_model": options["model"], "judge_effort": options["effort"]}


def run_judge(prompt: str, *, judge: dict | None = None, **kwargs):
    options = judge_options(judge)
    return run_codex(prompt, model=options["model"], effort=options["effort"], **kwargs)
