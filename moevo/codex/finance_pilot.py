"""Bounded FinQA harness-policy evolution using only a ChatGPT Codex account.

This is a finance integration pilot, not the seven-domain experiment or a SOTA
evaluation. It evolves prompts, tool availability, and a verification pass.
The model and reasoning effort stay fixed. Generated Python is never imported.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import tempfile
from pathlib import Path

from moevo.codex.client import DEFAULT_MODEL, require_chatgpt_login, run_codex
from moevo.core.types import Program
from moevo.search.database import ParetoDatabase

POLICY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "instructions": {"type": "string"},
        "verification": {"type": "string"},
        "verify": {"type": "boolean"},
        "use_tools": {"type": "boolean"},
    },
    "required": ["instructions", "verification", "verify", "use_tools"],
}
ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"program": {"type": "array", "items": {"type": "string"}}},
    "required": ["program"],
}
SEED = {
    "instructions": "Solve the user's task accurately using the supplied evidence.",
    "verification": "Check the proposed solution against the task and evidence. Correct any errors.",
    "verify": False,
    "use_tools": True,
}
OBJECTIVES = ["finance_accuracy", "token_efficiency"]


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_policy(policy: dict) -> dict:
    if set(policy) != set(SEED):
        raise ValueError("Policy fields differ from the fixed harness interface")
    for key in ("instructions", "verification"):
        if not isinstance(policy[key], str) or not 1 <= len(policy[key]) <= 6000:
            raise ValueError(f"Invalid policy text: {key}")
    for key in ("verify", "use_tools"):
        if not isinstance(policy[key], bool):
            raise ValueError(f"Policy {key} must be boolean")
    return policy


def load_data(root: Path) -> tuple[dict, dict, dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    train = {r["id"]: r for r in json.loads((root / "dataset/train.json").read_text())}
    dev = {r["id"]: r for r in json.loads((root / "dataset/dev.json").read_text())}
    # The official interpreter is a fixed arithmetic DSL, not Python execution.
    source = root / "code/evaluate/evaluate.py"
    tree = ast.parse(source.read_text())
    names = {"str_to_num", "process_row", "eval_program", "program_tokenization"}
    nodes: list[ast.stmt] = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    nodes += [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "all_ops" for t in n.targets)
    ]
    namespace: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return train, dev, namespace


def task_prompt(row: dict) -> str:
    # Deliberately exclude gold_inds, explanations, retrieved gold evidence and program.
    evidence = {k: row[k] for k in ("pre_text", "table", "post_text")}
    return (
        "Use only this financial report and question. Do not search for the benchmark or its answers.\n"
        + json.dumps(evidence)
        + "\nQuestion: "
        + row["qa"]["question"]
        + '\nReturn JSON {"program": [tokens]} in FinQA arithmetic DSL. Each step has '
        'four tokens: operator ending in "(", first argument, second argument, ")". '
        'End with "EOF". Operators: add, subtract, multiply, divide, exp, greater, '
        "table_max, table_min, table_sum, table_average. Use #0 for the first step result, "
        "#1 for the second, and const_100 or const_1 for constants. Table operations "
        'take the exact row label and "none". Example for (10-4)/2: '
        '["subtract(", "10", "4", ")", "divide(", "#0", "2", ")", "EOF"]. '
        "FinQA output convention: express ratios, growth rates, and percentage shares as decimal "
        "fractions (for example 0.25 for 25%), not multiplied by 100. Keep monetary quantities "
        "in the report's stated units. Percentage-point differences retain the input percentage units."
    )


def score_answer(text: str, row: dict, official: dict) -> tuple[float, str]:
    try:
        program = json.loads(text)["program"]
        if (
            not isinstance(program, list)
            or not program
            or len(program) > 201
            or program[-1] != "EOF"
            or any(not isinstance(t, str) or len(t) > 500 for t in program)
        ):
            return 0.0, "Invalid program structure"
        invalid, actual = official["eval_program"](program, row["table"])
        # Same equality and rounding as official evaluate_result; no tolerance tuning.
        score = float(not invalid and actual == row["qa"]["exe_ans"])
        return score, f"execution_valid={not bool(invalid)}, correct={bool(score)}"
    except (KeyError, ValueError, TypeError):
        return 0.0, "Malformed structured answer"


class Study:
    def __init__(self, root: Path, state: dict):
        self.root, self.state = root, state

    def save(self) -> None:
        write_json(self.root / "state.json", self.state)

    def call(self, prompt: str, *, schema: dict, tools: bool, purpose: str, cwd: Path):
        if self.state["calls_started"] >= self.state["config"]["max_calls"]:
            raise RuntimeError(
                "Call limit reached; results saved. Resume with an explicitly increased cap."
            )
        self.state["calls_started"] += 1
        call_id = self.state["calls_started"]
        self.save()  # Count failed/in-flight calls, including interrupted processes.
        print(f"call {call_id}/{self.state['config']['max_calls']}: {purpose}", flush=True)
        response = run_codex(
            prompt,
            cwd=cwd,
            model=self.state["config"]["model"],
            effort=self.state["config"]["effort"],
            schema=schema,
            tools=tools,
            timeout=self.state["config"]["timeout"],
            log_dir=self.root / "calls" / f"{call_id:04d}",
        )
        self.state["usage"].append({"call": call_id, "purpose": purpose, **response.usage})
        self.save()
        return response

    def evaluate(self, policy: dict, rows: list[dict], official: dict, phase: str) -> dict:
        identity = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:16]
        cache = self.state["evaluations"].setdefault(phase, {}).setdefault(identity, {})
        for row in rows:
            if row["id"] in cache:
                continue
            with tempfile.TemporaryDirectory(prefix="moevo-finance-task-") as workspace:
                prompt = policy["instructions"] + "\n\n" + task_prompt(row)
                response = self.call(
                    prompt,
                    schema=ANSWER_SCHEMA,
                    tools=policy["use_tools"],
                    purpose=f"{phase}:{row['id']}",
                    cwd=Path(workspace),
                )
                output_tokens = response.usage.get("output_tokens", 0)
                if policy["verify"]:
                    response = self.call(
                        prompt
                        + "\n\nProposed answer:\n"
                        + response.text
                        + "\n\n"
                        + policy["verification"],
                        schema=ANSWER_SCHEMA,
                        tools=policy["use_tools"],
                        purpose=f"{phase}:verify:{row['id']}",
                        cwd=Path(workspace),
                    )
                    output_tokens += response.usage.get("output_tokens", 0)
                score, feedback = score_answer(response.text, row, official)
                cache[row["id"]] = {
                    "score": score,
                    "feedback": feedback,
                    "response": response.text,
                    "output_tokens": output_tokens,
                }
                self.save()
        results = [cache[r["id"]] for r in rows]
        accuracy = sum(r["score"] for r in results) / len(results)
        tokens = sum(r["output_tokens"] for r in results) / len(results)
        # Fixed normalization; raw usage is retained. This is not another domain.
        return {
            "finance_accuracy": accuracy,
            "token_efficiency": 1 / (1 + tokens / 4000),
            "mean_output_tokens": tokens,
        }

    def run(self, train: dict, dev: dict, official: dict) -> None:
        config = self.state["config"]
        train_rows = [train[k] for k in self.state["train_ids"]]
        dev_rows = [dev[k] for k in self.state["dev_ids"]]
        if not self.state["programs"]:
            scores = self.evaluate(SEED, train_rows, official, "train")
            self.state["programs"].append(
                Program(id="seed", solution=json.dumps(SEED), metrics=scores).to_dict()
            )
            self.save()
        while len(self.state["programs"]) < config["generations"] + 1:
            index = len(self.state["programs"])
            pending = self.state.get("pending")
            if pending is None:
                database = ParetoDatabase(
                    OBJECTIVES,
                    population_size=8,
                    num_islands=1,
                    random_seed=config["seed"] + index,
                    selection=config["selection"],
                )
                for program in self.state["programs"]:
                    database.add(Program.from_dict(program), index)
                parent, context, _ = database.sample(num_context=2)
                parent_policy = json.loads(parent.solution)
                key = hashlib.sha256(
                    json.dumps(parent_policy, sort_keys=True).encode()
                ).hexdigest()[:16]
                feedback = {
                    k: {"score": v["score"], "feedback": v["feedback"]}
                    for k, v in self.state["evaluations"]["train"][key].items()
                }
                prompt = (
                    "Improve this reusable agent harness policy across task domains. The frozen model is GPT-6 Astra. "
                    "We optimize correctness and token efficiency separately. Change instructions, tool availability, "
                    "or whether to perform one verification pass. Do not hardcode task IDs, answers, benchmark paths, "
                    "or particular reports. No tools. Return exactly the policy JSON.\nParent:\n"
                    + parent.solution
                    + "\nTraining feedback:\n"
                    + json.dumps(feedback)
                    + "\nOther candidates:\n"
                    + json.dumps([{"policy": p.solution, "scores": p.metrics} for p in context])
                )
                with tempfile.TemporaryDirectory(prefix="moevo-policy-mutation-") as workspace:
                    response = self.call(
                        prompt,
                        schema=POLICY_SCHEMA,
                        tools=False,
                        purpose=f"mutation:{index}",
                        cwd=Path(workspace),
                    )
                pending = {
                    "policy": validate_policy(json.loads(response.text)),
                    "parent": parent.id,
                }
                self.state["pending"] = pending
                self.save()
            policy = pending["policy"]
            scores = self.evaluate(policy, train_rows, official, "train")
            program = Program(
                id=f"generation_{index}",
                solution=json.dumps(policy),
                metrics=scores,
                parent_id=pending["parent"],
                iteration=index,
            )
            self.state["programs"].append(program.to_dict())
            self.state.pop("pending", None)
            self.save()
            print(f"generation {index}: {scores}", flush=True)
        programs = [Program.from_dict(p) for p in self.state["programs"]]
        # Selected using training scores only. Eval does not change the choice.
        best = max(
            programs,
            key=lambda p: p.get_objective("finance_accuracy") * p.get_objective("token_efficiency"),
        )
        self.state["selected_id"] = best.id
        self.save()
        reports = {}
        for label, p in [("seed", programs[0]), ("evolved", best)]:
            reports[label] = {
                "harness_id": p.id,
                "scores": self.evaluate(json.loads(p.solution), dev_rows, official, "heldout_dev"),
            }
        self.state["status"] = "complete"
        self.save()
        report = {
            "study": "FinQA integration pilot, not a full-domain or SOTA result",
            "protocol": "full report context; official FinQA DSL execution scorer; tiny fixed train/dev subsets",
            "config": config,
            "cli_version": self.state["cli_version"],
            "train_ids": self.state["train_ids"],
            "heldout_dev_ids": self.state["dev_ids"],
            "reports": reports,
            "calls_started": self.state["calls_started"],
        }
        write_json(self.root / "report.json", report)
        write_json(self.root / "selected_policy.json", json.loads(best.solution))
        print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/finqa"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--effort", default="xhigh", choices=["low", "medium", "high", "xhigh", "max"]
    )
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--train-tasks", type=int, default=3)
    parser.add_argument("--eval-tasks", type=int, default=3)
    parser.add_argument("--max-calls", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection", choices=["pareto", "scalar"], default="pareto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train, dev, official = load_data(args.data)
    if (
        min(args.train_tasks, args.eval_tasks, args.max_calls, args.timeout) < 1
        or args.generations < 0
    ):
        parser.error(
            "Task counts, call limit, and timeout must be positive; generations nonnegative"
        )
    if args.train_tasks > len(train) or args.eval_tasks > len(dev):
        parser.error("Requested more tasks than available")
    config = {
        k: v for k, v in vars(args).items() if k not in {"data", "output", "resume", "dry_run"}
    }
    config["data_revision"] = json.loads((args.data / "manifest.json").read_text())["revision"]
    rng = random.Random(args.seed)
    train_ids = rng.sample(sorted(train), args.train_tasks)
    # Exclude reports seen during this pilot's evolution, as well as exact IDs.
    report_ids = {k.rsplit("-", 1)[0] for k in train_ids}
    eligible = [k for k in sorted(dev) if k.rsplit("-", 1)[0] not in report_ids]
    dev_ids = rng.sample(eligible, args.eval_tasks)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": config,
                    "train_ids": train_ids,
                    "dev_ids": dev_ids,
                    "no_model_calls": True,
                },
                indent=2,
            )
        )
        return
    cli_version = require_chatgpt_login()
    path = args.output.resolve() / "state.json"
    if path.exists():
        if not args.resume:
            parser.error("Output already contains a run; use --resume or a new directory")
        state = json.loads(path.read_text())
        if state["config"] != config or state["cli_version"] != cli_version:
            parser.error("Resume configuration/CLI changed; use the original settings")
    else:
        state = {
            "config": config,
            "cli_version": cli_version,
            "train_ids": train_ids,
            "dev_ids": dev_ids,
            "calls_started": 0,
            "programs": [],
            "evaluations": {},
            "usage": [],
            "status": "running",
        }
    study = Study(args.output.resolve(), state)
    study.save()
    try:
        study.run(train, dev, official)
    except Exception as exc:
        state["status"] = "interrupted"
        state["last_error"] = str(exc)
        study.save()
        raise


if __name__ == "__main__":
    main()
