"""Pinned tau text simulator with every model call routed through account Codex.

Run this module in the upstream Python 3.12 environment. The official simulator,
tools, task splits and reward implementation remain unchanged. JSON text tool
calls and an Astra user simulator make this an explicit protocol variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

from moevo.codex.client import account_environment, require_chatgpt_login, run_codex

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/external/tau3_bench"
MODEL = "gpt-6-astra"
SCHEMA = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments_json": {"type": "string"},
                },
                "required": ["name", "arguments_json"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["content", "tool_calls"],
    "additionalProperties": False,
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def forbid_api(*args, **kwargs):
    raise RuntimeError("Inference API transport is disabled; use account Codex")


class AccountTransport:
    def __init__(self, output: Path, timeout: int, instructions: str = ""):
        self.output = output
        self.instructions = instructions
        self.deadline = time.monotonic() + timeout
        self.calls = []

    def generate(self, model, messages, tools=None, tool_choice=None, call_name=None, **kwargs):
        from tau2.data_model.message import AssistantMessage, ToolCall

        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Account tau simulation exhausted its wall-clock budget")
        schemas = [tool.openai_schema for tool in tools or []]
        history = [message.model_dump(mode="json", exclude_none=True) for message in messages]
        payload = {"messages": history, "tools": schemas, "tool_choice": tool_choice}
        prompt = (
            "Continue the following simulated conversation as its assistant, following its "
            "system instructions and role. The tools below operate only on simulated benchmark "
            "state. Return one assistant message as JSON: content is the message text; "
            "tool_calls is an array of tool names and arguments_json strings encoding argument "
            "objects. Use an empty array for no tool calls. Do not invent tool results. "
            "The simulator will execute requested tools and return results on the next turn.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        if call_name == "agent_response" and self.instructions:
            prompt = self.instructions + "\n\n" + prompt
        number = len(self.calls)
        call_output = self.output / f"call-{number:03d}"
        write_json(call_output / "request.json", payload)
        with tempfile.TemporaryDirectory(prefix="moevo-tau-account-") as cwd:
            response = run_codex(
                prompt,
                cwd=Path(cwd),
                model=MODEL,
                effort="xhigh",
                tools=False,
                schema=SCHEMA,
                timeout=min(180, remaining),
                log_dir=call_output,
            )
        decoded = json.loads(response.text)
        calls = []
        available = {tool.name for tool in tools or []}
        for i, call in enumerate(decoded["tool_calls"]):
            arguments = json.loads(call["arguments_json"])
            if call["name"] not in available or not isinstance(arguments, dict):
                raise ValueError("Invalid simulated tool name or argument object")
            calls.append(ToolCall(id=f"call-{number}-{i}", name=call["name"], arguments=arguments))
        record = {
            "call": number,
            "role": call_name,
            "requested_model": model,
            "actual_model": MODEL,
            "effort": "xhigh",
            "usage": response.usage,
            "duration_s": response.duration_s,
            "response": decoded,
        }
        self.calls.append(record)
        write_json(call_output / "response.json", record)
        return AssistantMessage.text(
            content=decoded["content"],
            tool_calls=calls or None,
            usage=response.usage,
            generation_time_seconds=response.duration_s,
        )


def install_transport(transport: AccountTransport):
    import litellm
    import tau2.utils.llm_utils as llm

    original = llm.generate
    replaced = []
    # tau2 imports its agent, user and evaluators on initialization. Replace
    # their imported-by-value references, plus the source for later imports.
    for name, module in list(sys.modules.items()):
        if (
            name.startswith("tau2.")
            and module is not None
            and getattr(module, "generate", None) is original
        ):
            module.__dict__["generate"] = transport.generate
            replaced.append(name)
    if not {"tau2.agent.llm_agent", "tau2.user.user_simulator"}.issubset(replaced):
        raise RuntimeError("Upstream model call sites changed; account routing is incomplete")
    llm.completion = forbid_api
    litellm.completion = forbid_api
    litellm.acompletion = forbid_api
    return replaced


def validate_controls(task, domain: str, output: Path):
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.registry import registry

    constructor = registry.get_env_constructor(domain)
    env = constructor()
    state = task.initial_state
    history = list(state.message_history or []) if state else []
    env.set_state(
        initialization_data=state.initialization_data if state else None,
        initialization_actions=state.initialization_actions if state else None,
        message_history=history,
    )
    trajectory = list(history)
    if task.evaluation_criteria is None or not task.evaluation_criteria.actions:
        raise ValueError("This DB-control pilot requires a task with gold actions")
    for i, action in enumerate(task.evaluation_criteria.actions):
        call = ToolCall(
            id=f"control-{i}",
            name=action.name,
            arguments=action.arguments,
            requestor=action.requestor,
        )
        cls = AssistantMessage if action.requestor == "assistant" else UserMessage
        trajectory.append(cls(role=action.requestor, content="", tool_calls=[call]))
        reply = env.get_response(call)
        if reply.error:
            raise ValueError(f"Gold action failed: {action.name}")
        trajectory.append(reply)
    positive = EnvironmentEvaluator.calculate_reward(constructor, task, trajectory)
    negative = EnvironmentEvaluator.calculate_reward(constructor, task, history)
    report = {
        "positive": positive.model_dump(mode="json"),
        "negative": negative.model_dump(mode="json"),
        "scope": "Native end-state evaluator: replay gold actions versus no-op",
    }
    write_json(output / "controls.json", report)
    if positive.reward != 1 or negative.reward != 0:
        raise ValueError("Gold/no-op controls did not distinguish completion from failure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--policy-file", type=Path)
    args = parser.parse_args()
    policy = json.loads(args.policy_file.read_text()) if args.policy_file else None
    if policy and (policy["model"] != MODEL or policy["authentication"] != "codex_chatgpt_account"):
        raise ValueError("Tau policy must use account Astra")
    clean = account_environment()
    os.environ.clear()
    os.environ.update(clean)
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["LOG_LEVEL"] = "ERROR"
    from loguru import logger
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.run import get_tasks, run_single_task

    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    output = args.output.resolve() / "tau3_retail"
    version = require_chatgpt_login() if not args.controls_only else None
    tasks = get_tasks("retail", task_split_name="train")
    random.Random(args.seed).shuffle(tasks)
    signature = {
        "model": MODEL,
        "effort": "xhigh",
        "user_model": MODEL,
        "source": json.loads((SOURCE / "source_manifest.json").read_text())["revision"],
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed,
        "split": "train",
        "timeout": args.timeout,
        "max_steps": 50,
        "shared_policy": policy,
    }
    write_json(output / "panel.json", {"signature": signature, "ids": [t.id for t in tasks[:32]]})
    for task in tasks[: args.count]:
        task_output = output / f"task-{task.id}"
        task_output.mkdir(parents=True, exist_ok=True)
        cached = task_output / "report.json"
        if cached.exists():
            if json.loads(cached.read_text())["signature"] != signature:
                raise ValueError("Cached tau configuration changed")
            continue
        try:
            validate_controls(task, "retail", task_output)
            if args.controls_only:
                continue
            transport = AccountTransport(
                task_output / "calls", args.timeout, policy["instructions"] if policy else ""
            )
            routes = install_transport(transport)
            config = TextRunConfig(
                domain="retail",
                agent="llm_agent",
                user="user_simulator",
                llm_agent=MODEL,
                llm_user=MODEL,
                llm_args_agent={},
                llm_args_user={},
                max_steps=50,
                timeout=args.timeout,
                auto_review=False,
                review_model=MODEL,
            )
            simulation = run_single_task(
                config,
                task,
                seed=args.seed,
                evaluation_type=EvaluationType.ALL,
                save_dir=task_output,
                auto_review=False,
            )
            write_json(task_output / "simulation.json", simulation.model_dump(mode="json"))
            if not transport.calls or simulation.reward_info is None:
                raise ValueError("Simulation produced no model calls or reward")
            report = {
                "dataset": "tau3-bench retail",
                "task_id": task.id,
                "signature": signature,
                "cli_version": version,
                "reward": simulation.reward_info.model_dump(mode="json"),
                "termination": simulation.termination_reason.value,
                "model_calls": len(transport.calls),
                "e2e_smoke_passed": True,
                "verified_headroom": False,
                "account_routes": routes,
                "protocol": "Native text simulator and reward; account Astra agent/user; JSON tool transport; 50-step cap",
            }
            write_json(cached, report)
            print(json.dumps(report), flush=True)
        except Exception as exc:
            write_json(task_output / "error.json", {"error": str(exc), "scored": False})
            raise


if __name__ == "__main__":
    main()
