"""E2E public AutomationBench task with Astra account + official state assertions."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from moevo.codex.automation_adapter import load_official, make_state
from moevo.codex.client import CodexError, require_chatgpt_login, run_codex
from moevo.codex.finance_pilot import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        default="operations",
        choices=["sales", "marketing", "operations", "finance", "hr", "support"],
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--max-tools", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = Path("data/external/automationbench").resolve()
    official = load_official(source)
    dataset = official["dataset"](args.domain)
    task = dict(dataset[args.index])
    state = make_state(task, official)
    before = official["partial_credit"](state)
    if official["success"](state):
        raise ValueError(
            "This task is already passed by the no-op control; choose an action task for smoke validation"
        )
    output = (
        args.output
        or Path(f"results/domain_validation/automationbench_v2/{args.domain}-{args.index:03d}")
    ).resolve()
    if (output / "report.json").exists():
        print((output / "report.json").read_text())
        return
    output.mkdir(parents=True, exist_ok=True)
    version = require_chatgpt_login()
    prompt = "Complete this simulated workflow using only the benchmark MCP tools. The tools affect only an isolated synthetic business world.\n"
    prompt += "\n\n".join(m["role"].upper() + ": " + m["content"] for m in task["prompt"])
    with (
        tempfile.TemporaryDirectory(prefix="moevo-automation-agent-") as workspace,
        tempfile.TemporaryDirectory(prefix="moevo-automation-control-") as control,
    ):
        control_path = Path(control) / "control.json"
        write_json(
            control_path,
            {
                "source": str(source),
                "task": task,
                "state_output": str(output / "agent_state.json"),
                "max_tool_calls": args.max_tools,
            },
        )
        response = run_codex(
            prompt,
            cwd=Path(workspace),
            tools=False,
            timeout=args.timeout,
            mcp_server=[
                sys.executable,
                "-m",
                "moevo.codex.automation_server",
                "--control",
                str(control_path),
            ],
            approved_mcp_tools=("search_tools", "execute_tool"),
            log_dir=output / "codex",
        )
    agent_state = json.loads((output / "agent_state.json").read_text())
    if not any(t["tool"] == "execute" and "result" in t for t in agent_state["trace"]):
        raise CodexError("No benchmark action completed; E2E validation is incomplete")
    state["world"] = official["world"](**agent_state["world"])
    partial = official["partial_credit"](state)
    success = official["success"](state)
    report = {
        "dataset": "AutomationBench public",
        "domain": args.domain,
        "task_id": task["example_id"],
        "task_name": state["info"].get("task_name"),
        "source": json.loads((source / "source_manifest.json").read_text()),
        "model": "gpt-6-astra",
        "effort": "xhigh",
        "cli_version": version,
        "toolset": "zapier meta-tools via MCP",
        "max_tool_calls": args.max_tools,
        "tool_calls": agent_state["tool_calls"],
        "noop_partial_credit": before,
        "partial_credit": partial,
        "task_completed_correctly": success,
        "e2e_smoke_passed": agent_state["tool_calls"] > 0,
        "headroom_confirmed": False,
        "official_private_leaderboard_comparable": False,
        "usage": response.usage,
        "duration_s": response.duration_s,
        "response": response.text,
        "assertions": state["_assertion_results"],
    }
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"response", "assertions", "source"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
