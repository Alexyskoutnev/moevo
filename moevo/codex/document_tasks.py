"""Legal and GDPval artifact tasks with account-only rubric judges."""

from __future__ import annotations

import concurrent.futures
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from moevo.codex.domain_tasks import ROOT, result, solve
from moevo.codex.finance_pilot import write_json
from moevo.codex.judging import judge_metadata, run_judge

IMAGE = "docker.io/library/moevo-doc-runtime:20260915"
VERDICT = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
    },
    "required": ["reasoning", "verdict"],
    "additionalProperties": False,
}


def extract(directory: Path, output: Path):
    directory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
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
            "--tmpfs",
            "/tmp:rw,size=256m",
            "--mount",
            f"type=bind,source={directory},target=/files,readonly",
            "--mount",
            f"type=bind,source={ROOT / 'experiments/runtime/extract_deliverables.py'},target=/extract.py,readonly",
            IMAGE,
            "python",
            "/extract.py",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    files = json.loads(completed.stdout)
    write_json(output, files)
    return files


def legal_verdict(
    task: str, text: str, title: str, criterion: str, output: Path, *, judge: dict | None = None
):
    template = (
        ROOT / "data/external/harvey_lab/lab_core/evaluation/prompts/rubric_criterion.txt"
    ).read_text()
    prompt = template.format(
        task_description=task, agent_output=text, criterion_title=title, match_criteria=criterion
    )
    with tempfile.TemporaryDirectory(prefix="moevo-legal-judge-") as cwd:
        response = run_judge(
            prompt, judge=judge, cwd=Path(cwd), schema=VERDICT, timeout=240, log_dir=output
        )
    verdict = json.loads(response.text)
    if verdict["verdict"] not in {"pass", "fail"} or not isinstance(verdict.get("reasoning"), str):
        raise ValueError("Invalid legal judge verdict")
    verdict.update(judge_metadata(judge))
    verdict["judge_usage"] = response.usage
    write_json(output / "verdict.json", verdict)
    return verdict


def legal(policy: dict, output: Path, image: str):
    source = ROOT / "data/external/harvey_lab"
    task_id = "banking-finance/extract-lien-and-debt-information-from-ucc-filings"
    task = source / "tasks" / task_id
    config = json.loads((task / "task.json").read_text())
    for label, text, expected in [
        ("positive", "The contract value is USD 100.", "pass"),
        ("negative", "No analysis submitted.", "fail"),
    ]:
        verdict = legal_verdict(
            "Extract the contract value",
            text,
            "Value",
            "PASS if the response states that the contract value is USD 100; otherwise FAIL.",
            output / "controls" / label,
            judge=policy,
        )
        if verdict["verdict"] != expected:
            raise ValueError("Legal rubric judge failed transport control")
    workspace = output / "workspace"
    shutil.copytree(task / "documents", workspace / "documents")
    (workspace / "output").mkdir()
    (workspace / "output").chmod(0o777)
    response = solve(
        policy,
        config["instructions"]
        + "\nInputs are in /workspace/documents. Save all requested deliverables in /workspace/output. Pandoc and python-docx are installed.",
        output,
        IMAGE,
    )
    files = {f["filename"]: f for f in extract(workspace / "output", output / "extracted.json")}

    def grade_one(item):
        i, criterion = item
        scoped = []
        for filename in criterion["deliverables"]:
            file = files.get(filename)
            scoped.append(
                f"## Agent Output: {filename}\n"
                + (json.dumps(file, ensure_ascii=False) if file else "(File not found)")
            )
        verdict = legal_verdict(
            config["title"],
            "\n\n".join(scoped),
            criterion["title"],
            criterion["match_criteria"],
            output / "grade" / f"criterion-{i:03d}",
            judge=policy,
        )
        return {"id": criterion["id"], **verdict}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        grades = list(pool.map(grade_one, enumerate(config["criteria"])))
    passed = sum(g["verdict"] == "pass" for g in grades)
    score = float(bool(grades) and passed == len(grades))
    grade = {
        "all_pass": bool(score),
        "n_passed": passed,
        "n_criteria": len(grades),
        "criteria_results": grades,
    }
    write_json(output / "grade.json", grade)
    return result(
        response,
        score,
        grade,
        task_id,
        "Official LAB criterion prompt and all-pass rule; account judge variant; DOCX extracted with pandoc",
        **judge_metadata(policy),
        criterion_pass_rate=passed / len(grades),
        controls_scope="Rubric transport; not an independent legal accuracy audit",
        artifact_count=len(files),
        runtime_image=IMAGE,
    )


def gdp_grade(prompt: str, files: list, criteria: list, output: Path, *, judge: dict | None = None):
    from moevo.eval.evaluators.gdpval_judge import JUDGE_SYSTEM_PROMPT

    schema = {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "met": {"type": "boolean"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["id", "met", "reasoning"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["criteria"],
        "additionalProperties": False,
    }
    request = (
        JUDGE_SYSTEM_PROMPT
        + "\nFor this transport, return each criterion's integer ID, met boolean and reasoning. A negative criterion is met when its penalized behavior occurs. Do not total the score; the caller will calculate it. Return every supplied ID exactly once.\n"
    )
    request += json.dumps(
        {
            "task": prompt,
            "actual_files": files,
            "rubric": [{"id": i, **c} for i, c in enumerate(criteria)],
        },
        ensure_ascii=False,
    )
    with tempfile.TemporaryDirectory(prefix="moevo-gdp-judge-") as cwd:
        response = run_judge(
            request, judge=judge, cwd=Path(cwd), schema=schema, timeout=600, log_dir=output
        )
    grades = json.loads(response.text)["criteria"]
    if (
        not isinstance(grades, list)
        or any(
            not isinstance(g, dict)
            or type(g.get("id")) is not int
            or type(g.get("met")) is not bool
            or not isinstance(g.get("reasoning"), str)
            for g in grades
        )
        or sorted(g["id"] for g in grades) != list(range(len(criteria)))
    ):
        raise ValueError("GDPval judge omitted or duplicated criteria")
    awarded = sum(criteria[g["id"]]["score"] for g in grades if g["met"])
    possible = sum(max(0, c["score"]) for c in criteria)
    if possible <= 0:
        raise ValueError("GDPval rubric has no positive points")
    grade = {
        "raw_points": awarded,
        "positive_points": possible,
        "score": max(0, min(1, awarded / possible)),
        "criteria": grades,
        **judge_metadata(judge),
        "judge_usage": response.usage,
    }
    write_json(output / "grade.json", grade)
    return grade


def gdp(policy: dict, output: Path, image: str):
    import pyarrow.parquet as pq

    source = ROOT / "data/raw/gdpval"
    rows = pq.read_table(next(source.rglob("*.parquet"))).to_pylist()
    # A fixed artifact-based task, chosen before any model answer is observed.
    row = rows[0]
    control_criteria = [{"score": 2, "criterion": "The file contains both 2 and 3."}]
    for label, text, expected in [("positive", "2 and 3", 1), ("negative", "No answer", 0)]:
        grade = gdp_grade(
            "Write both numbers 2 and 3.",
            [{"filename": "answer.txt", "text": text}],
            control_criteria,
            output / "controls" / label,
            judge=policy,
        )
        if grade["score"] != expected:
            raise ValueError("GDPval account judge failed controls")
    workspace = output / "workspace"
    refs = workspace / "reference_files"
    refs.mkdir(parents=True)
    for filename in row["reference_files"]:
        origin = (source / filename).resolve()
        if not origin.is_relative_to(source.resolve() / "reference_files"):
            raise ValueError("Unexpected GDPval reference path")
        shutil.copyfile(origin, refs / origin.name)
    (workspace / "output").mkdir()
    (workspace / "output").chmod(0o777)
    response = solve(
        policy,
        row["prompt"]
        + "\nInput files are in /workspace/reference_files. Save all final deliverables in /workspace/output.",
        output,
        IMAGE,
    )
    files = extract(workspace / "output", output / "extracted.json")
    grade = gdp_grade(
        row["prompt"], files, json.loads(row["rubric_json"]), output / "grade", judge=policy
    )
    return result(
        response,
        grade["score"],
        grade,
        row["task_id"],
        "Public GDPval weighted rubric and actual artifact extraction; account judge variant; not expert pairwise win rate",
        **judge_metadata(policy),
        artifact_count=len(files),
        runtime_image=IMAGE,
        visual_layout_evaluated=False,
        controls_scope="Rubric transport and point calculation",
    )
