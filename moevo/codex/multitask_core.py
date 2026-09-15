"""Exact-task development adapters for six downloaded benchmark sources.

The fixture adapters remain untouched. Inventory and structural preflight make
no model calls. Native control checks are opt-in; DSBench's rubric controls run
only during evaluate(), through the account judge configured in the policy.
Unknown prior exposure is conservatively marked exposed, never fresh held-out.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = {
    "finqa": "finance",
    "bizfinbench2": "finance",
    "genebench_pro": "science",
    "amo": "mathematics",
    "travelplanner": "planning",
    "dsbench": "data_analysis",
}
DATA_FOLDERS = {**{key: key for key in DOMAINS}, "amo": "amo_bench"}
IMAGE = "docker.io/library/moevo-or-runtime:20260915"
DEVELOPMENT_ONLY = {
    "finqa": {"SLB/2011/page_41.pdf-1", "ADI/2009/page_49.pdf-1"},
    "bizfinbench2": {"numeric-0656", "numeric-0000"},
    "genebench_pro": {"wf_selection", "carrier_cnv_pseudogene_residual_risk"},
    "amo": {"p-subset-0036", "p-subset-0000"},
    "travelplanner": {"train-12", "train-0"},
    "dsbench": {"00000034/question9"},
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


@lru_cache(maxsize=4096)
def _file_digest(path: Path, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _hash_file(path: Path) -> str:
    stat = path.stat()
    return _file_digest(path.resolve(), stat.st_size, stat.st_mtime_ns)


def _inside(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(parent.resolve()) or not resolved.is_file():
        raise ValueError(f"Invalid or missing public input path: {path}")
    return resolved


def _source(benchmark: str, root: Path) -> dict:
    folder = root / "data/raw" / DATA_FOLDERS[benchmark]
    path = folder / ("manifest.json" if benchmark == "finqa" else "_moevo_manifest.json")
    data = _json(path)
    info = {
        "dataset_revision": data["revision"],
        "repository": data["repository"],
        "manifest_sha256": _hash_file(path),
        "manifest_path": str(path.relative_to(root)),
        "split": {"finqa": "train", "travelplanner": "train"}.get(benchmark, "public development"),
        "exposure_status": "unknown_or_previously_used_conservatively_exposed",
    }
    external = root / "data/external" / DATA_FOLDERS[benchmark] / "source_manifest.json"
    if external.exists():
        reference = _json(external)
        info["reference_repository"] = reference["repository"]
        info["reference_revision"] = reference["revision"]
        info["reference_manifest_sha256"] = _hash_file(external)
    if benchmark == "dsbench":
        info["task_metadata_sha256"] = _hash_file(
            root / "data/external/dsbench/data_analysis/data.json"
        )
        info["judge_prompt_source_sha256"] = _hash_file(
            root / "data/external/dsbench/data_analysis/compute_answer.py"
        )
    return info


def _rows(benchmark: str, root: Path) -> dict[str, dict]:
    folder = root / "data/raw" / DATA_FOLDERS[benchmark]
    if benchmark == "finqa":
        rows = _json(folder / "dataset/train.json")
        pairs = [(row["id"], row) for row in rows]
    elif benchmark == "bizfinbench2":
        rows = [
            json.loads(line)
            for line in (folder / "en/financial_quantitative_computation_en.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        pairs = [(f"numeric-{index:04d}", row) for index, row in enumerate(rows)]
    elif benchmark == "genebench_pro":
        pairs = [
            (path.parent.name, _json(path))
            for path in sorted((folder / "problems").glob("*/eval_config.json"))
        ]
    elif benchmark == "amo":
        import pyarrow.parquet as pq

        paths = sorted(folder.rglob("*.parquet"))
        if len(paths) != 1:
            raise ValueError("AMO currently requires exactly one pinned parquet shard")
        rows = [
            row
            for row in pq.read_table(paths[0]).to_pylist()
            if row["answer_type"] != "description"
        ]
        if any(row["answer_type"] not in {"number", "set", "variable"} for row in rows):
            raise ValueError("AMO parser-only subset contains an unsupported answer type")
        pairs = [(f"p-subset-{index:04d}", row) for index, row in enumerate(rows)]
    elif benchmark == "travelplanner":
        import csv

        with (folder / "train.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in ("days", "people_number", "visiting_city_number", "budget"):
                row[key] = int(row[key])
            for key in ("date", "local_constraint", "annotated_plan", "reference_information"):
                row[key] = ast.literal_eval(row[key])
        pairs = [(f"train-{index}", row) for index, row in enumerate(rows)]
    elif benchmark == "dsbench":
        source = root / "data/external/dsbench/data_analysis/data.json"
        groups = [
            ast.literal_eval(line) for line in source.read_text().splitlines() if line.strip()
        ]
        pairs = []
        for group in groups:
            if len(group["questions"]) != len(group["answers"]):
                raise ValueError("DSBench question/answer alignment failed")
            for index, name in enumerate(group["questions"]):
                pairs.append(
                    (
                        f"{group['id']}/{name}",
                        {**group, "question_index": index, "question_name": name},
                    )
                )
    else:
        raise ValueError(f"Unsupported multi-task benchmark: {benchmark}")
    if not pairs or len({task_id for task_id, _ in pairs}) != len(pairs):
        raise ValueError(f"Empty or duplicate task IDs: {benchmark}")
    return dict(pairs)


def _lookup(benchmark: str, task_id: str, root: Path) -> dict:
    if benchmark not in DOMAINS:
        raise ValueError(f"Unsupported multi-task benchmark: {benchmark}")
    rows = _rows(benchmark, root)
    if task_id not in rows:
        raise KeyError(f"Unknown {benchmark} task ID: {task_id}")
    return rows[task_id]


def _biz_prompt(row: dict) -> str:
    parts = []
    for message in row["messages"]:
        if message["role"] == "assistant":
            continue
        content = message["content"]
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise ValueError("BizFin numeric adapter supports text inputs only")
            content = "\n".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise ValueError("Malformed BizFin prompt")
        parts.append(message["role"].upper() + ": " + content)
    if not parts:
        raise ValueError("Missing BizFin public prompt")
    return (
        "\n\n".join(parts)
        + "\nReturn the requested JSON; use decimal rather than scientific notation."
    )


def _public(
    benchmark: str, task_id: str, row: dict, root: Path
) -> tuple[str, list[tuple[Path, str]]]:
    """Public text and allowlisted source files; no labels/configs reach the solver."""
    from moevo.codex.finance_pilot import task_prompt

    inputs: list[tuple[Path, str]] = []
    if benchmark == "finqa":
        return task_prompt(row), inputs
    if benchmark == "bizfinbench2":
        return _biz_prompt(row), inputs
    if benchmark == "amo":
        return row[
            "prompt"
        ] + "\nUse the benchmark run tool to check calculations or constructions.", inputs
    if benchmark == "genebench_pro":
        problem = root / "data/raw/genebench_pro/problems" / task_id
        for filename in row["data_files"]:
            relative = Path(filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.parts[0] != "data_files"
            ):
                raise ValueError("Invalid GeneBench public input path")
            inputs.append(
                (_inside(problem / relative, problem / "data_files"), relative.as_posix())
            )
        return (
            row["task"]
            + "\nInputs are in /workspace/data_files. Return the requested JSON in your final answer.",
            inputs,
        )
    if benchmark == "travelplanner":
        # Reference information is intentionally generated separately from the row
        # because annotated_plan contains the answer and must stay grader-side.
        return (
            row["query"]
            + "\nThe sole-planning reference information is in /workspace/reference_information.json. Return only a JSON object with a plan array. Each day must contain days (integer), current_city, transportation, breakfast, attraction, lunch, dinner, accommodation. Use exact reference names; meals and accommodation are 'Name, City', attractions 'Name, City;'. Use '-' for unused fields. Intercity current_city is 'from Origin to Destination'. Flights use 'Flight Number: F..., from Origin to Destination, Departure Time: HH:MM, Arrival Time: HH:MM'. Respect every applicable constraint.",
            inputs,
        )
    folder = root / "data/raw/dsbench/processed/data" / row["id"]
    base = root / "data/raw/dsbench/processed/data"
    if not folder.resolve().is_relative_to(base.resolve()) or not folder.is_dir():
        raise ValueError("Invalid DSBench input group")
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in {".xlsx", ".xls", ".csv"} or path.name == "introduction.txt":
            inputs.append((_inside(path, folder), path.name))
    question = _inside(folder / (row["question_name"] + ".txt"), folder)
    inputs.append((question, "question.txt"))
    if len(inputs) < 2:
        raise ValueError("DSBench question has no analysis inputs")
    return (
        "Solve this data-analysis task using the supplied workbook/data files and introduction in /workspace. Use the benchmark run tool to inspect files and calculate the answer. Return a clear final answer, including the option letter when multiple choice.\n\n"
        + question.read_text(),
        inputs,
    )


def inventory(benchmarks: list[str] | None = None, *, root: Path = ROOT) -> list[dict]:
    """Return task metadata only, with group IDs suitable for leakage-aware splits."""
    result = []
    for benchmark in benchmarks if benchmarks is not None else DOMAINS:
        if benchmark not in DOMAINS:
            raise ValueError(f"Unsupported multi-task benchmark: {benchmark}")
        source = _source(benchmark, root)
        for task_id, row in _rows(benchmark, root).items():
            prompt, inputs = _public(benchmark, task_id, row, root)
            public = {"prompt": prompt, "inputs": {name: _hash_file(path) for path, name in inputs}}
            if benchmark == "travelplanner":
                public["reference_information"] = row["reference_information"]
            digest = _digest(public)
            if benchmark == "finqa":
                group = "/".join(row["filename"].split("/")[:2])
            elif benchmark == "dsbench":
                group = row["id"]
            elif benchmark == "amo":
                group = str(row["question_id"])
            elif benchmark == "bizfinbench2":
                group = digest
            else:
                group = task_id
            result.append(
                {
                    "id": task_id,
                    "benchmark": benchmark,
                    "domain": DOMAINS[benchmark],
                    "group_id": f"{benchmark}:{group}",
                    "content_sha256": digest,
                    "exposed": True,
                    "adapter_ready": True,
                    "development_only": task_id in DEVELOPMENT_ONLY.get(benchmark, set()),
                    "source": {
                        **source,
                        "row_sha256": _digest(row),
                        "input_sha256": public["inputs"],
                    },
                }
            )
    return result


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("moevo_reference_" + _hash_file(path), path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load pinned reference grader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _biz_grade(row: dict, prediction: str, output: Path, root: Path) -> dict:
    source = root / "data/external/bizfinbench2"
    parser = _module(source / "utils/JsonPaser.py").JsonPaser
    path = source / "benchmark_code/BizFinBench.v2/eval_financial_quantitative_computation.py"
    # Keep official function bodies; provide its one local import explicitly so
    # concurrent benchmark adapters never alter sys.modules['utils'].
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
    namespace = {"JsonPaser": parser}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    output.mkdir(parents=True, exist_ok=True)
    record = {**copy.deepcopy(row), "predict_result": prediction}
    target = output / "prediction.jsonl"
    target.write_text(json.dumps(record) + "\n")
    namespace["evaluation"](str(target))
    record = json.loads(target.read_text())
    if "error" in record["eval_result"]:
        raise ValueError("BizFin reference grader error: " + record["eval_result"]["error"])
    return {"score": record["score"], "grade": record["eval_result"]}


def _native_grade(
    benchmark: str, row: dict, prediction: Any, output: Path, image: str, root: Path
) -> dict:
    from moevo.codex.finance_pilot import load_data, score_answer, write_json

    if benchmark == "finqa":
        _, _, official = load_data(root / "data/raw/finqa")
        score, feedback = score_answer(prediction, row, official)
        result = {"score": score, "feedback": feedback}
    elif benchmark == "bizfinbench2":
        result = _biz_grade(row, prediction, output, root)
    elif benchmark == "genebench_pro":
        grader = _module(root / "data/raw/genebench_pro/reference_grader.py")
        result = grader.evaluate(row, prediction)
    elif benchmark in {"amo", "travelplanner"}:
        if root.resolve() != ROOT.resolve():
            raise ValueError("Container graders require repository source root")
        from experiments.validate_domains import offline_grade

        result = offline_grade(
            "travel" if benchmark == "travelplanner" else "amo",
            row,
            prediction,
            output.resolve(),
            image,
        )
    else:
        raise ValueError("DSBench uses a live rubric judge, not a native offline grader")
    value = result.get("score")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("Reference grader returned an invalid normalized score")
    write_json(output / "grade.json", result)
    return result


def _control_answers(benchmark: str, row: dict, root: Path) -> tuple[Any, Any]:
    if benchmark == "finqa":
        from moevo.codex.finance_pilot import load_data

        _, _, official = load_data(root / "data/raw/finqa")
        return json.dumps(
            {"program": official["program_tokenization"](row["qa"]["program"])}
        ), '{"program":[]}'
    if benchmark == "bizfinbench2":
        return json.dumps(
            {"answer": row["choices"][0]["message"]["content"][0]["text"]}
        ), '{"answer":"no numerical answer"}'
    if benchmark == "genebench_pro":
        return {"answer": row["ground_truth"]}, {"answer": {}}
    if benchmark == "amo":
        return row["answer"], "No answer submitted"
    if benchmark == "travelplanner":
        return row["annotated_plan"][1], []
    raise ValueError("No offline controls for rubric-only DSBench")


def preflight(
    benchmark: str,
    task_id: str,
    output: Path | None = None,
    image: str = IMAGE,
    *,
    native_controls: bool = False,
    root: Path = ROOT,
) -> dict:
    """Verify the exact task; never sample a replacement for an invalid annotation."""
    row = _lookup(benchmark, task_id, root)
    prompt, inputs = _public(benchmark, task_id, row, root)
    source = _source(benchmark, root)
    public = {"prompt": prompt, "inputs": {name: _hash_file(path) for path, name in inputs}}
    if benchmark == "travelplanner":
        public["reference_information"] = row["reference_information"]
    report = {
        "benchmark": benchmark,
        "task_id": task_id,
        "content_sha256": _digest(public),
        "input_files": len(inputs),
        "source": source,
        "structural_preflight_passed": True,
        "native_controls_passed": None,
        "model_calls": 0,
        "e2e_validated": False,
    }
    if native_controls and benchmark != "dsbench":
        if output is None:
            raise ValueError("Native controls require a task-specific output directory")
        positive, negative = _control_answers(benchmark, row, root)
        pos = _native_grade(benchmark, row, positive, output / "positive_control", image, root)
        neg = _native_grade(benchmark, row, negative, output / "negative_control", image, root)
        passed = (
            bool(pos.get("passed", pos["score"] == 1))
            and neg["score"] == 0
            and not bool(neg.get("passed", False))
        )
        report.update(
            {"controls": {"positive": pos, "negative": neg}, "native_controls_passed": passed}
        )
        if not passed:
            from moevo.codex.finance_pilot import write_json

            write_json(output / "preflight.json", report)
            raise ValueError(
                f"Reference controls failed for exact {benchmark}/{task_id}; task not replaced"
            )
    elif benchmark == "dsbench":
        from experiments.validate_domains import ds_judge_prompt

        question = next(path for path, name in inputs if name == "question.txt").read_text()
        ds_judge_prompt(question, str(row["answers"][row["question_index"]]), "No answer submitted")
        report["judge_controls_status"] = "pending_live_account_judge_checks"
    if output is not None:
        from moevo.codex.finance_pilot import write_json

        write_json(output / "preflight.json", report)
    return report


def evaluate(benchmark: str, task_id: str, policy: dict, output: Path, image: str = IMAGE) -> dict:
    """Execute and grade one exact task using Astra and policy-pinned rubric judges."""
    from moevo.codex.domain_tasks import result, solve
    from moevo.codex.finance_pilot import write_json
    from moevo.codex.judging import judge_metadata, judge_options

    for key, expected in {
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "authentication": "codex_chatgpt_account",
    }.items():
        if policy.get(key) != expected:
            raise ValueError(f"The current account task runtime requires {key}={expected}")
    judge_options(policy)
    if not isinstance(policy.get("instructions"), str) or not policy["instructions"].strip():
        raise ValueError("Missing shared harness instructions")
    for key in ("timeout_seconds", "max_tool_calls"):
        if type(policy.get(key)) is not int or policy[key] < 1:
            raise ValueError(f"Invalid {key}")
    output = output.resolve()
    workspace = output / "workspace"
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError("Refusing to reuse a nonempty solver workspace")
    check = preflight(benchmark, task_id, output, image, native_controls=True)
    row = _lookup(benchmark, task_id, ROOT)
    prompt, inputs = _public(benchmark, task_id, row, ROOT)
    workspace.mkdir(parents=True, exist_ok=True)
    for source, relative in inputs:
        target = workspace / relative
        if target.is_symlink() or not target.resolve().is_relative_to(workspace.resolve()):
            raise ValueError("Unsafe solver input target")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if benchmark == "travelplanner":
        write_json(workspace / "reference_information.json", row["reference_information"])
    controls = check.get("controls", {})
    if benchmark == "dsbench":
        from experiments.validate_domains import ds_grade

        question = (workspace / "question.txt").read_text()
        gold = str(row["answers"][row["question_index"]])
        controls = {
            "positive": ds_grade(question, gold, gold, output / "positive_control", judge=policy),
            "negative": ds_grade(
                question, gold, "No answer submitted", output / "negative_control", judge=policy
            ),
        }
        if controls["positive"]["score"] != 1 or controls["negative"]["score"] != 0:
            raise ValueError("DSBench account judge controls failed")
    response = solve(policy, prompt, output, image)
    parse_error = None
    if benchmark == "dsbench":
        scored = ds_grade(question, gold, response.text, output / "grade", judge=policy)
    else:
        prediction: Any = response.text
        if benchmark == "genebench_pro":
            try:
                prediction = json.loads(response.text)
            except json.JSONDecodeError:
                prediction, parse_error = {}, "invalid_json"
            if not isinstance(prediction, dict):
                prediction, parse_error = {}, "invalid_json_object"
        elif benchmark == "travelplanner":
            from experiments.validate_domains import decode_plan

            prediction, parse_error = decode_plan(response.text)
        scored = _native_grade(benchmark, row, prediction, output / "grade", image, ROOT)
    report = result(
        response,
        scored["score"],
        scored,
        task_id,
        "Exact requested development task; native grader or explicitly pinned account rubric variant",
        benchmark=benchmark,
        content_sha256=check["content_sha256"],
        source=check["source"],
        controls=controls,
        parse_error=parse_error,
        strict_pass=scored.get("passed", scored["score"] == 1),
        **(judge_metadata(policy) if benchmark == "dsbench" else {}),
    )
    write_json(output / "report.json", report)
    return report
