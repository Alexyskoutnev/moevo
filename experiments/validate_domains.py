"""Account-only E2E validation for downloaded math, planning, and analysis tasks.

This runner screens development tasks. It does not evolve on final holdouts or
claim that a smoke-test score demonstrates headroom.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from moevo.codex.client import account_environment, require_chatgpt_login
from moevo.codex.container_runtime import solve_in_container
from moevo.codex.finance_pilot import write_json
from moevo.codex.headroom import assess_headroom

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "docker.io/library/moevo-or-runtime:20260915"
SOURCES = {"amo": "amo_bench", "travel": "travelplanner", "dsbench": "dsbench"}


def offline_grade(dataset: str, row: dict, prediction, output: Path, image: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "input.json", {"dataset": dataset, "row": row, "prediction": prediction})
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=4g",
        "--cpus=2",
        "--pids-limit=256",
        "--user=1000:1000",
        "--tmpfs",
        "/tmp:rw,size=256m,mode=1777",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--mount",
        f"type=bind,source={output},target=/control,readonly",
        "--mount",
        f"type=bind,source={ROOT / 'data/external' / SOURCES[dataset]},target=/benchmark,readonly",
        "--mount",
        f"type=bind,source={ROOT / 'experiments/runtime/grade_domain.py'},target=/grade.py,readonly",
    ]
    if dataset == "travel":
        command += [
            "--mount",
            f"type=bind,source={ROOT / 'data/raw/travelplanner/processed/database'},target=/benchmark/database,readonly",
        ]
    completed = subprocess.run(
        [*command, image, "python", "/grade.py"], capture_output=True, text=True, timeout=180
    )
    (output / "stderr.txt").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError("Offline grader failed: " + completed.stderr[-3000:])
    result = json.loads(completed.stdout)
    write_json(output / "grade.json", result)
    return result


def tasks(dataset: str) -> list[dict]:
    if dataset == "amo":
        import pyarrow.parquet as pq

        path = next((ROOT / "data/raw/amo_bench").rglob("*.parquet"))
        return [r for r in pq.read_table(path).to_pylist() if r["answer_type"] != "description"]
    if dataset == "travel":
        with (ROOT / "data/raw/travelplanner/train.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in ["days", "people_number", "visiting_city_number", "budget"]:
                row[key] = int(row[key])
            for key in ["date", "local_constraint", "annotated_plan", "reference_information"]:
                row[key] = ast.literal_eval(row[key])
        return rows
    path = ROOT / "data/external/dsbench/data_analysis/data.json"
    groups = [ast.literal_eval(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        {**group, "question_index": index, "question_name": name}
        for group in groups
        for index, name in enumerate(group["questions"])
    ]


def ds_judge_prompt(question: str, answer: str, prediction: str) -> str:
    # Reuse the original prompt expression, while replacing only its API transport.
    path = ROOT / "data/external/dsbench/data_analysis/compute_answer.py"
    function = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "evaluate_prediction"
    )
    assignment = next(
        n
        for n in function.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "prompt" for t in n.targets)
    )
    return eval(
        compile(ast.Expression(assignment.value), str(path), "eval"),
        {"question": question, "answer": answer, "prediction": prediction},
    )


def ds_grade(
    question: str, gold: str, prediction: str, output: Path, *, judge: dict | None = None
) -> dict:
    from moevo.codex.judging import judge_metadata, run_judge

    with tempfile.TemporaryDirectory(prefix="moevo-ds-judge-") as cwd:
        result = run_judge(
            ds_judge_prompt(question, gold, prediction),
            judge=judge,
            cwd=Path(cwd),
            timeout=180,
            log_dir=output,
        )
    verdict = result.text.strip().lower()
    if verdict not in {"true", "false", "flase"}:
        raise ValueError("DSBench judge did not return a Boolean verdict")
    return {
        "score": float(verdict == "true"),
        "judge_response": result.text,
        "judge_usage": result.usage,
        **judge_metadata(judge),
        "protocol": "Official judge prompt; account model replaces original GPT-4o judge",
    }


def prompt_and_inputs(dataset: str, row: dict, workspace: Path) -> str:
    workspace.mkdir(parents=True, exist_ok=True)
    if dataset == "amo":
        return (
            row["prompt"] + "\nUse the benchmark run tool to check calculations or constructions."
        )
    if dataset == "travel":
        write_json(workspace / "reference_information.json", row["reference_information"])
        return (
            row["query"] + "\nThe sole-planning reference information is in "
            "/workspace/reference_information.json. Use the benchmark run tool to inspect it. "
            "Return only a JSON object with a plan array. Each day must contain: days (integer), "
            "current_city, transportation, breakfast, attraction, lunch, dinner, accommodation. "
            "Use exact names from the reference information: meals and accommodation as 'Name, City', "
            "attractions as 'Name, City;Name, City;'. Use '-' for an unused field. "
            "For intercity travel current_city is 'from Origin to Destination'; otherwise it is the city. "
            "Flights use 'Flight Number: F..., from Origin to Destination, Departure Time: HH:MM, "
            "Arrival Time: HH:MM'. Preserve applicable budget, hotel minimum stay, cuisine, "
            "room, transportation and route constraints."
        )
    folder = ROOT / "data/raw/dsbench/processed/data" / row["id"]
    for path in folder.iterdir():
        if path.suffix.lower() in {".xlsx", ".xls", ".csv"} or path.name == "introduction.txt":
            shutil.copy2(path, workspace / path.name)
    question = (folder / (row["question_name"] + ".txt")).read_text()
    (workspace / "question.txt").write_text(question)
    return (
        "Solve this data-analysis task using the supplied workbook/data files and introduction "
        "in /workspace. Use the benchmark run tool to inspect the files and calculate the answer. "
        "Do not infer the answer from a filename. Return a clear final answer to this question, "
        "including the option letter when it is multiple choice.\n\n" + question
    )


def decode_plan(response: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip())
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return [], "invalid_json"
    plan = result.get("plan", []) if isinstance(result, dict) else result
    fields = {
        "days",
        "current_city",
        "transportation",
        "breakfast",
        "attraction",
        "lunch",
        "dinner",
        "accommodation",
    }
    if not isinstance(plan, list) or any(
        not isinstance(day, dict)
        or not fields.issubset(day)
        or not isinstance(day["days"], int)
        or any(not isinstance(day[key], str) for key in fields - {"days"})
        for day in plan
    ):
        return [], "invalid_plan_schema"
    return plan, None


def validate(
    dataset: str,
    count: int,
    seed: int,
    timeout: int,
    root: Path,
    image: str,
    version: str | None,
    controls_only: bool,
):
    rows = tasks(dataset)
    indices = random.Random(seed).sample(range(len(rows)), min(count, len(rows)))
    source = json.loads(
        (ROOT / "data/external" / SOURCES[dataset] / "source_manifest.json").read_text()
    )
    signature = {
        "dataset": dataset,
        "image_id": image,
        "timeout_seconds": timeout,
        "model": "gpt-6-astra",
        "effort": "xhigh",
        "seed": seed,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "grader_sha256": hashlib.sha256(
            (ROOT / "experiments/runtime/grade_domain.py").read_bytes()
        ).hexdigest(),
    }
    reports = []
    for index in indices:
        row = rows[index]
        output = root / dataset / f"task-{index:04d}"
        output.mkdir(parents=True, exist_ok=True)
        cached = output / "report.json"
        if cached.exists() and not controls_only:
            report = json.loads(cached.read_text())
            if report["run_signature"] != signature:
                raise ValueError("Cached run differs; choose a fresh --output directory")
            reports.append(report)
            continue
        try:
            if dataset in {"amo", "travel"}:
                gold = row["answer"] if dataset == "amo" else row["annotated_plan"][1]
                wrong = "No answer submitted" if dataset == "amo" else []
                positive = offline_grade(dataset, row, gold, output / "positive_control", image)
                negative = offline_grade(dataset, row, wrong, output / "negative_control", image)
            else:
                question = (
                    ROOT
                    / "data/raw/dsbench/processed/data"
                    / row["id"]
                    / (row["question_name"] + ".txt")
                ).read_text()
                gold = str(row["answers"][row["question_index"]])
                if controls_only:
                    write_json(
                        output / "input_inventory.json",
                        {
                            "competition": row["id"],
                            "question": row["question_name"],
                            "requires_account_judge": True,
                        },
                    )
                    continue
                positive = ds_grade(question, gold, gold, output / "positive_control")
                negative = ds_grade(
                    question, gold, "No answer submitted", output / "negative_control"
                )
            write_json(output / "controls.json", {"positive": positive, "negative": negative})
            if positive["score"] != 1 or negative["score"] != 0:
                raise ValueError("Official grading controls failed; this is not model headroom")
            if controls_only:
                print(
                    json.dumps({"dataset": dataset, "index": index, "controls": "passed"}),
                    flush=True,
                )
                continue
            prompt = prompt_and_inputs(dataset, row, output / "workspace")
            response = solve_in_container(
                prompt, output / "workspace", output / "agent", image=image, timeout=timeout
            )
            parse_error = None
            if dataset == "dsbench":
                grade = ds_grade(question, gold, response.text, output / "prediction_judge")
            else:
                prediction = response.text
                if dataset == "travel":
                    prediction, parse_error = decode_plan(response.text)
                grade = offline_grade(dataset, row, prediction, output / "prediction", image)
            report = {
                "dataset": dataset,
                "index": index,
                "run_signature": signature,
                "source": source,
                "cli_version": version,
                "score": grade["score"],
                "grade": grade,
                "parse_error": parse_error,
                "response": response.text,
                "usage": response.usage,
                "duration_s": response.duration_s,
                "e2e_smoke_passed": True,
                "semantic_failure_audited": False,
                "protocol": {
                    "amo": "Official parser-only P subset",
                    "travel": "Sole-planning train; official constraint evaluator",
                    "dsbench": "Official assets and judge prompt; Astra judge variant",
                }[dataset],
            }
            if dataset == "dsbench":
                report["competition"] = row["id"]
                report["question"] = row["question_name"]
            write_json(cached, report)
            reports.append(report)
            print(
                json.dumps({"dataset": dataset, "index": index, "score": grade["score"]}),
                flush=True,
            )
        except Exception as exc:
            write_json(
                output / "infrastructure_error.json",
                {"error": repr(exc), "run_signature": signature},
            )
            print(
                json.dumps({"dataset": dataset, "index": index, "infrastructure_error": str(exc)}),
                flush=True,
            )
            # Do not turn transport, controls, or grader errors into benchmark zeros.
            raise
    if reports:
        summary = {
            "dataset": dataset,
            "indices": indices,
            **assess_headroom([r["score"] for r in reports]),
        }
        write_json(root / dataset / "summary.json", summary)
        print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=[*SOURCES, "health", "tau"], default=list(SOURCES)
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/domain_validation/domains_v1"
    )
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.count <= 32 or not 60 <= args.timeout <= 1800:
        raise ValueError("Development screening allows 1–32 tasks and 60–1800 seconds per task")
    image = IMAGE
    if any(dataset in SOURCES for dataset in args.datasets):
        inspected = subprocess.check_output(
            ["docker", "image", "inspect", IMAGE], text=True, timeout=30
        )
        image = json.loads(inspected)[0]["Id"]
    version = (
        require_chatgpt_login() if not args.controls_only or "health" in args.datasets else None
    )
    failures = []
    for dataset in args.datasets:
        try:
            if dataset == "tau":
                command = [
                    str(ROOT / "data/external/tau3_bench/.venv/bin/python"),
                    "-m",
                    "moevo.codex.tau_pilot",
                    "--count",
                    str(args.count),
                    "--seed",
                    str(args.seed),
                    "--timeout",
                    str(args.timeout),
                    "--output",
                    str(args.output.resolve()),
                ]
                if args.controls_only:
                    command.append("--controls-only")
                subprocess.run(command, cwd=ROOT, env=account_environment(), check=True)
                continue
            if dataset == "health":
                from moevo.codex.health_pilot import validate_health

                validate_health(
                    args.count,
                    args.seed,
                    args.timeout,
                    args.output.resolve(),
                    version,
                    args.controls_only,
                )
                continue
            validate(
                dataset,
                args.count,
                args.seed,
                args.timeout,
                args.output.resolve(),
                image,
                version,
                args.controls_only,
            )
        except Exception as exc:
            failures.append({"dataset": dataset, "error": str(exc)})
    if failures:
        raise RuntimeError(json.dumps(failures))


if __name__ == "__main__":
    main()
