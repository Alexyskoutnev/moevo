"""Measure a fixed Astra baseline on FinQA train; no evolution or API-key calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path

from moevo.codex.client import DEFAULT_MODEL, CodexError, require_chatgpt_login, run_codex
from moevo.codex.finance_pilot import (
    ANSWER_SCHEMA,
    SEED,
    load_data,
    score_answer,
    task_prompt,
    write_json,
)
from moevo.codex.headroom import assess_headroom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path("results/domain_validation/finqa"))
    args = parser.parse_args()
    train, _, official = load_data(Path("data/raw/finqa"))
    if not 1 <= args.tasks <= 128 or args.timeout < 1:
        parser.error("Choose 1–128 screening tasks and a positive timeout")
    # A fixed 128-task ordering allows extending a pilot without changing earlier IDs.
    ids = random.Random(args.seed).sample(sorted(train), 128)[: args.tasks]
    cli_version = require_chatgpt_login()
    config = {
        "dataset": "finqa",
        "model": DEFAULT_MODEL,
        "effort": "xhigh",
        "seed": args.seed,
        "timeout": args.timeout,
        "cli_version": cli_version,
        "policy": SEED,
        "split": "train",
        "purpose": "headroom_screen_only",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"config": config, "results": {}, "calls_started": 0}
    )
    if state["config"] != config:
        parser.error("Screen settings changed; use a fresh output directory")
    # Check the official interpreter against every selected released gold program
    # and a deliberately invalid answer before spending model calls.
    for task_id in ids:
        row = train[task_id]
        gold = json.dumps({"program": official["program_tokenization"](row["qa"]["program"])})
        if score_answer(gold, row, official)[0] != 1:
            raise ValueError(f"Official gold program does not match its label: {task_id}")
        if score_answer('{"program": ["invalid(", "1", "1", ")", "EOF"]}', row, official)[0] != 0:
            raise ValueError("Invalid-answer control incorrectly received credit")
    for index, task_id in enumerate(ids):
        if task_id in state["results"]:
            continue
        row = train[task_id]
        prompt = SEED["instructions"] + "\n\n" + task_prompt(row)
        call = state["calls_started"] + 1
        state["calls_started"] = call
        write_json(state_path, state)
        print(f"FinQA {index + 1}/{len(ids)}: {task_id}", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix="moevo-finqa-screen-") as workspace:
                response = run_codex(
                    prompt,
                    cwd=Path(workspace),
                    schema=ANSWER_SCHEMA,
                    tools=True,
                    timeout=args.timeout,
                    log_dir=args.output / "calls" / f"{call:04d}",
                )
            score, feedback = score_answer(response.text, row, official)
            result = {
                "score": score,
                "feedback": feedback,
                "response": response.text,
                "usage": response.usage,
                "duration_s": response.duration_s,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "grader_controls_passed": True,
            }
            state["results"][task_id] = result
            print(
                json.dumps({"task_id": task_id, "score": score, "feedback": feedback}), flush=True
            )
        except CodexError as error:
            state.setdefault("infrastructure_errors", []).append(
                {"task_id": task_id, "call": call, "error": str(error)}
            )
            write_json(state_path, state)
            raise
        write_json(state_path, state)
        scores = [state["results"][k]["score"] for k in ids if k in state["results"]]
        write_json(
            args.output / "report.json",
            {
                "config": config,
                "task_ids": ids,
                "e2e_smoke_passed": bool(scores),
                "headroom": assess_headroom(scores),
                "calls_started": state["calls_started"],
                "failure_review": "Pending; do not admit merely because the observed score is below 100%.",
            },
        )


if __name__ == "__main__":
    main()
