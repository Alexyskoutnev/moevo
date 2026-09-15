"""Real development-task adapters sharing one account-Astra harness policy."""

from __future__ import annotations

import copy
import importlib.util
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

from moevo.codex.client import run_codex
from moevo.codex.container_runtime import solve_in_container
from moevo.codex.finance_pilot import write_json

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load reference scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def solve(policy: dict, prompt: str, output: Path, image: str):
    return solve_in_container(
        policy["instructions"] + "\n\n" + prompt,
        output / "workspace",
        output / "agent",
        image=image,
        timeout=policy["timeout_seconds"],
        max_tools=policy["max_tool_calls"],
    )


def result(response, score: float, grade: dict, task_id: str, protocol: str, **extra):
    return {
        "score": score,
        "grade": grade,
        "task_id": task_id,
        "prediction": response.text,
        "usage": response.usage,
        "duration_s": response.duration_s,
        "e2e_validated": True,
        "protocol": protocol,
        **extra,
    }


def basic(name: str, policy: dict, output: Path, image: str):
    from experiments.run_submission import fixture

    prompt, _, grade, metadata = fixture(name, policy["seed"], output, image)
    response = solve(policy, prompt, output, image)
    scored = grade(response.text, "prediction")
    return result(
        response,
        scored["score"],
        scored,
        metadata.pop("task_id"),
        "Official reference grader; one development task; shared account-Astra policy",
        **metadata,
    )


def bizfin(policy: dict, output: Path, image: str):
    source = ROOT / "data/external/bizfinbench2"
    sys.path.insert(0, str(source))
    scorer = load_module(
        "bizfin_submission_numeric",
        source / "benchmark_code/BizFinBench.v2/eval_financial_quantitative_computation.py",
    )
    with (
        ROOT / "data/raw/bizfinbench2/en/financial_quantitative_computation_en.jsonl"
    ).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    index = random.Random(policy["seed"]).sample(range(len(rows)), 128)[0]
    row = rows[index]

    def grade(prediction: str, label: str):
        record = copy.deepcopy(row)
        record["predict_result"] = prediction
        path = output / f"{label}.jsonl"
        path.write_text(json.dumps(record) + "\n")
        scorer.evaluation(str(path))
        record = json.loads(path.read_text())
        if "error" in record["eval_result"]:
            raise ValueError(record["eval_result"]["error"])
        return {"score": record["score"], "grade": record["eval_result"]}

    gold = row["choices"][0]["message"]["content"][0]["text"]
    controls = {
        "positive": grade(json.dumps({"answer": gold}), "positive"),
        "negative": grade('{"answer":"no numerical answer"}', "negative"),
    }
    if controls["positive"]["score"] != 1 or controls["negative"]["score"] != 0:
        raise ValueError("Numeric reference controls failed")
    parts = []
    for message in row["messages"]:
        if message["role"] == "assistant":
            continue
        content = message["content"]
        if isinstance(content, list):
            content = "\n".join(p["text"] for p in content if p.get("type") == "text")
        parts.append(message["role"].upper() + ": " + content)
    response = solve(
        policy,
        "\n\n".join(parts)
        + "\nReturn the requested JSON; use decimal rather than scientific notation.",
        output,
        image,
    )
    scored = grade(response.text, "prediction")
    return result(
        response,
        scored["score"],
        scored,
        f"numeric-{index:04d}",
        "Official BizFinBench.v2 numeric grader; English subset",
        controls=controls,
    )


def gene(policy: dict, output: Path, image: str):
    source = ROOT / "data/raw/genebench_pro"
    problem = source / "problems/wf_selection"
    config = json.loads((problem / "eval_config.json").read_text())
    grader = load_module("gene_submission_reference", source / "reference_grader.py")
    positive = grader.evaluate(config, {"answer": config["ground_truth"]})
    negative = grader.evaluate(config, {"answer": {}})
    if not positive["passed"] or negative["passed"]:
        raise ValueError("Gene reference controls failed")
    workspace = output / "workspace"
    for filename in config["data_files"]:
        origin = (problem / filename).resolve()
        if not origin.is_relative_to((problem / "data_files").resolve()):
            raise ValueError("Invalid public input path")
        target = workspace / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
    response = solve(
        policy,
        config["task"]
        + "\nInputs are in /workspace/data_files. Return the requested JSON in your final answer.",
        output,
        image,
    )
    try:
        prediction = json.loads(response.text)
    except json.JSONDecodeError:
        prediction = {}
    if not isinstance(prediction, dict):
        prediction = {}
    scored = grader.evaluate(config, prediction)
    write_json(output / "controls.json", {"positive": positive, "negative": negative})
    return result(
        response,
        scored["score"],
        scored,
        "wf_selection",
        "Public 10-case release; official composite grader",
        strict_pass=scored["passed"],
    )


def analysis(policy: dict, output: Path, image: str):
    from experiments.validate_domains import ds_grade, prompt_and_inputs, tasks

    rows = tasks("dsbench")
    index = random.Random(policy["seed"]).sample(range(len(rows)), 1)[0]
    row = rows[index]
    prompt = prompt_and_inputs("dsbench", row, output / "workspace")
    question = (output / "workspace/question.txt").read_text()
    gold = str(row["answers"][row["question_index"]])
    positive = ds_grade(question, gold, gold, output / "positive_control")
    negative = ds_grade(question, gold, "No answer submitted", output / "negative_control")
    if positive["score"] != 1 or negative["score"] != 0:
        raise ValueError("DSBench account judge controls failed")
    response = solve(policy, prompt, output, image)
    scored = ds_grade(question, gold, response.text, output / "grade")
    return result(
        response,
        scored["score"],
        scored,
        f"{row['id']}/{row['question_name']}",
        "Official data and judge prompt; account-Astra judge variant",
    )


def health(policy: dict, output: Path, image: str):
    from moevo.codex.health_pilot import DATA, controls, development_panel, grade_response

    with (DATA / "healthbench_professional_eval.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in rows:
        row["conversation"] = row["conversation"]["messages"]
    controls(output, {"policy": policy, "protocol": "account-Astra per-criterion public rubric"})
    panel, _ = development_panel(rows, policy["seed"])
    row = rows[panel[0]]
    prompt = policy["instructions"] + "\nRespond to the final user message:\n\n"
    prompt += "\n\n".join(f"{m['role']}: {m['content']}" for m in row["conversation"])
    with tempfile.TemporaryDirectory(prefix="moevo-health-submission-") as cwd:
        response = run_codex(
            prompt,
            cwd=Path(cwd),
            model=policy["model"],
            effort=policy["reasoning_effort"],
            timeout=policy["timeout_seconds"],
            log_dir=output / "agent",
        )
    scored = grade_response(row, response.text, output / "grade")
    return result(
        response,
        scored["raw_rubric_score"],
        scored,
        row["id"],
        "Public HealthBench scorer; account-Astra solver/judge; text only; external protocol variant",
        strict_pass=scored["all_criteria_passed"],
    )


def travel(policy: dict, output: Path, image: str):
    from experiments.validate_domains import decode_plan, offline_grade, prompt_and_inputs, tasks

    rows = tasks("travel")
    indices = list(range(len(rows)))
    random.Random(policy["seed"]).shuffle(indices)
    checks = []
    # This is adapter validation, not accuracy estimation. Select the first
    # valid released annotation, recording rejected controls before any solver call.
    for index in indices[:10]:
        row = rows[index]
        positive = offline_grade(
            "travel", row, row["annotated_plan"][1], output / f"gold-{index}", image
        )
        checks.append({"index": index, "score": positive["score"]})
        write_json(output / "annotation_checks.json", checks)
        if positive["score"] == 1:
            break
    else:
        raise ValueError("No valid released annotation in the 10-task control panel")
    negative = offline_grade("travel", row, [], output / "negative_control", image)
    if negative["score"] != 0:
        raise ValueError("Travel no-plan control passed")
    prompt = prompt_and_inputs("travel", row, output / "workspace")
    response = solve(policy, prompt, output, image)
    prediction, parse_error = decode_plan(response.text)
    scored = offline_grade("travel", row, prediction, output / "grade", image)
    return result(
        response,
        scored["score"],
        scored,
        f"train-{index}",
        "Official sole-planning constraints; fixture selected for valid annotation; not a representative score",
        parse_error=parse_error,
        annotation_checks=checks,
    )


def automation(policy: dict, output: Path, image: str):
    from moevo.codex.automation_adapter import load_official, make_state

    official = load_official(ROOT / "data/external/automationbench")
    task = dict(official["dataset"]("operations")[0])
    state = make_state(task, official)
    before = official["partial_credit"](state)
    if official["success"](state):
        raise ValueError("Automation task already passes before acting")
    prompt = (
        policy["instructions"]
        + "\nComplete this simulated workflow using only benchmark MCP tools. These tools affect only a synthetic business world.\n"
    )
    prompt += "\n\n".join(m["role"].upper() + ": " + m["content"] for m in task["prompt"])
    with (
        tempfile.TemporaryDirectory(prefix="moevo-automation-submission-") as cwd,
        tempfile.TemporaryDirectory(prefix="moevo-automation-control-") as control,
    ):
        config = Path(control) / "control.json"
        write_json(
            config,
            {
                "source": str(ROOT / "data/external/automationbench"),
                "task": task,
                "state_output": str(output / "agent_state.json"),
                "max_tool_calls": policy["max_tool_calls"],
            },
        )
        response = run_codex(
            prompt,
            cwd=Path(cwd),
            timeout=policy["timeout_seconds"],
            mcp_server=[
                sys.executable,
                "-m",
                "moevo.codex.automation_server",
                "--control",
                str(config),
            ],
            approved_mcp_tools=("search_tools", "execute_tool"),
            log_dir=output / "agent",
        )
    captured = json.loads((output / "agent_state.json").read_text())
    if not any(t["tool"] == "execute" and "result" in t for t in captured["trace"]):
        raise ValueError("No simulated action executed")
    state["world"] = official["world"](**captured["world"])
    score = official["partial_credit"](state)
    success = official["success"](state)
    grade = {
        "partial_credit": score,
        "strict_pass": success,
        "assertions": state["_assertion_results"],
    }
    return result(
        response,
        score,
        grade,
        task["example_id"],
        "Official public simulated-world assertions; known task-ID inconsistency remains under audit",
        noop_partial_credit=before,
        strict_pass=success,
        positive_control_validated=False,
        e2e_validated=False,
        validation_note="Real execution and grading completed; positive-control/ID audit still required",
    )


HANDLERS = {
    "finqa": lambda p, o, i: basic("finqa", p, o, i),
    "amo": lambda p, o, i: basic("amo", p, o, i),
    "bizfinbench2": bizfin,
    "genebench_pro": gene,
    "dsbench": analysis,
    "healthbench_professional": health,
    "travelplanner": travel,
    "automationbench": automation,
}
