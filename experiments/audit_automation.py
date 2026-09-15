"""Check a recorded AutomationBench run without changing its native score."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from moevo.codex.automation_adapter import load_official, make_state
from moevo.codex.finance_pilot import write_json

ROOT = Path(__file__).resolve().parents[1]


def audit(attempt: Path) -> dict:
    os.environ["AUTOMATIONBENCH_STRICT_ASSERTIONS"] = "1"
    official = load_official(ROOT / "data/external/automationbench")
    task = dict(official["dataset"]("operations")[0])
    captured = json.loads((attempt / "agent_state.json").read_text())
    state = make_state(task, official)
    negative = official["partial_credit"](state)
    state["world"] = official["world"](**captured["world"])
    actual = official["partial_credit"](state)
    # Synthetic positive fixture for the same assertions, kept outside agent inputs.
    # This is deliberately not represented as a reachable task oracle.
    synthetic = copy.deepcopy(captured["world"])
    assertions = state["info"]["assertions"]
    expected_ids = set()
    for assertion in assertions:
        if assertion["type"] == "asana_action_exists":
            action = assertion["action_key"]
            params = assertion["params"]
            synthetic["asana"]["actions"][action] = [
                {
                    "id": "synthetic-control-" + action,
                    "action_key": action,
                    "params": params,
                }
            ]
            if "task_id" in params:
                expected_ids.add(params["task_id"])
    state["world"] = official["world"](**synthetic)
    positive = official["partial_credit"](state)
    if negative != 0 or positive != 1:
        raise ValueError("Automation assertion controls failed")
    created_ids = [a["id"] for a in captured["world"]["asana"]["actions"].get("create_task", [])]
    report = {
        "task_id": task["example_id"],
        "native_score_unchanged": actual,
        "negative_noop_score": negative,
        "synthetic_positive_score": positive,
        "strict_assertion_errors": False,
        "expected_task_ids": sorted(expected_ids),
        "created_task_ids": created_ids,
        "id_contract_mismatch": bool(expected_ids - set(created_ids)),
        "scope": "Controls prove native assertions distinguish passing/failing states; synthetic positive is not a reachable oracle. Fixed assertion IDs disagree with generated tool IDs. Score remains under audit.",
    }
    write_json(attempt / "assertion_audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.attempt.resolve()), indent=2))


if __name__ == "__main__":
    main()
