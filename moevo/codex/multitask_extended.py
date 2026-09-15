"""Explicit-task adapters for the extended SuperHarness benchmark collection.

Inventory/preflight are offline and never dispatch a model. Readiness describes
implemented protocol support and local task assets, not measured accuracy or a
successful runtime control. Unknown historical exposure is conservatively true.
The existing one-fixture adapters are deliberately left untouched.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

from moevo.codex.client import account_environment, run_codex
from moevo.codex.domain_tasks import result, solve
from moevo.codex.finance_pilot import write_json
from moevo.codex.judging import judge_metadata, judge_options

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "docker.io/library/moevo-or-runtime:20260915"
DOMAINS = {
    "gdpval": "professional work",
    "harvey_lab": "legal",
    "healthbench_professional": "medicine",
    "putnambench": "mathematics",
    "oragentbench": "operations research",
    "terminal_bench_2": "software terminal",
    "tau3_bench": "customer service",
}
TEXT_ARTIFACTS = {".docx", ".xlsx", ".pdf", ".pptx", ".txt", ".md", ".csv", ".json"}
DEVELOPMENT_ONLY = {
    "gdpval": {"83d10b06-26d1-4636-a32c-23f92c57f30b"},
    "harvey_lab": {"banking-finance/extract-lien-and-debt-information-from-ucc-filings"},
    "healthbench_professional": {"d52fb2405d6f6c57e1e1472bfe5be65e"},
    "putnambench": {"putnam_2010_a4"},
    "oragentbench": {"industrial_water_reuse_blending"},
    "terminal_bench_2": {"regex-log"},
    "tau3_bench": {"retail/1"},
}


class UnsupportedTaskError(ValueError):
    """The exact task is known but its native integration is not ready."""


UnsupportedTask = UnsupportedTaskError


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(root: Path, relative: str, *, exists: bool = True) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
        raise ValueError(f"Invalid task-relative path: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Task path escaped its source: {relative}")
    if exists and not path.exists():
        raise FileNotFoundError(path)
    return path


def _source(benchmark: str, root: Path | None = None) -> tuple[Path, dict]:
    area = "raw" if benchmark in {"gdpval", "healthbench_professional"} else "external"
    source = (root or ROOT) / "data" / area / benchmark
    filename = "_moevo_manifest.json" if area == "raw" else "source_manifest.json"
    manifest = json.loads((source / filename).read_text())
    return source, {
        key: manifest[key]
        for key in (
            "repository",
            "revision",
            "archive_sha256",
        )
        if key in manifest
    }


def _gdp_rows(source: Path) -> list[dict]:
    import pyarrow.parquet as pq

    paths = sorted(source.glob("data/*.parquet"))
    if not paths:
        raise FileNotFoundError("GDPval data parquet is missing")
    rows = [row for path in paths for row in pq.read_table(path).to_pylist()]
    return rows


def _health_rows(source: Path) -> list[dict]:
    with (source / "healthbench_professional_eval.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in rows:
        if isinstance(row["conversation"], dict):
            row["conversation"] = row["conversation"]["messages"]
    return rows


def _tau_rows(source: Path) -> tuple[list[dict], dict]:
    root = source / "data/tau2/domains/retail"
    return json.loads((root / "tasks.json").read_text()), json.loads(
        (root / "split_tasks.json").read_text()
    )


def _legal_documents(source: Path, task: Path, row: dict) -> Path:
    configured = row.get("docs_dir")
    docs = (task / configured).resolve() if configured else task / "documents"
    # Upstream allows shared docs_dir relative to task; restrict it to this release.
    if not docs.is_relative_to(source.resolve()):
        raise ValueError("LAB docs_dir escaped the pinned release")
    if not docs.is_dir():
        raise FileNotFoundError(f"LAB documents directory missing: {docs}")
    return docs


def _base(benchmark: str, task_id: str, content: Any, group: str, source: dict) -> dict:
    return {
        "id": task_id,
        "benchmark": benchmark,
        "domain": DOMAINS[benchmark],
        "group_id": group,
        "content_sha256": _hash(content),
        "content_hash_scope": "task specification, grading record and pinned source identity; selected input bytes separately hashed at preflight",
        "exposed": True,
        "exposure_status": "unknown_or_previously_used",
        "development_only": task_id in DEVELOPMENT_ONLY.get(benchmark, set()),
        "source": source,
        "adapter_ready": True,
        "exclusion_reason": None,
        "readiness_stage": "static support; runtime controls and model execution not yet run",
    }


def _exclude(row: dict, reason: str) -> dict:
    row.update(adapter_ready=False, exclusion_reason=reason)
    return row


def _gdp_groups(rows: list[dict], checksums: dict) -> dict[str, str]:
    """Connected components prevent shared-reference tasks crossing partitions."""
    parents = {row["task_id"]: row["task_id"] for row in rows}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    owner = {}
    for row in rows:
        for path in row["reference_files"]:
            digest = checksums.get(path, path)
            if digest in owner:
                a, b = find(row["task_id"]), find(owner[digest])
                parents[max(a, b)] = min(a, b)
            else:
                owner[digest] = row["task_id"]
    return {key: "reference-family/" + find(key) for key in parents}


def _inventory_one(benchmark: str, root: Path | None = None) -> list[dict]:
    source, provenance = _source(benchmark, root)
    records = []
    if benchmark == "gdpval":
        rows = _gdp_rows(source)
        checksum = json.loads((source / "_moevo_manifest.json").read_text()).get("files", {})
        groups = _gdp_groups(rows, checksum)
        for task in rows:
            row = _base(
                benchmark, task["task_id"], [task, provenance], groups[task["task_id"]], provenance
            )
            row.update(
                occupation=task.get("occupation"), sector=task.get("sector"), original_split="train"
            )
            if any(
                Path(name).suffix.lower() not in TEXT_ARTIFACTS
                for name in task["deliverable_files"]
            ):
                _exclude(
                    row,
                    "Declared output format needs non-text/visual validation outside this artifact rubric protocol",
                )
            elif any(not (source / path).is_file() for path in task["reference_files"]):
                _exclude(row, "Required reference file missing")
            elif len({Path(path).name for path in task["reference_files"]}) != len(
                task["reference_files"]
            ):
                _exclude(row, "Reference basenames collide in the flat workspace mapping")
            records.append(row)
    elif benchmark == "harvey_lab":
        for path in sorted((source / "tasks").rglob("task.json")):
            task = json.loads(path.read_text())
            task_id = path.parent.relative_to(source / "tasks").as_posix()
            family = re.sub(r"/scenario-[^/]+$", "", task_id)
            try:
                docs = _legal_documents(source, path.parent, task)
                if task.get("docs_dir"):
                    family = "shared-docs/" + docs.relative_to(source).as_posix()
                reason = None
            except (ValueError, FileNotFoundError) as exc:
                reason = str(exc)
            row = _base(benchmark, task_id, [task, provenance], family, provenance)
            criteria = task.get("criteria", [])
            if reason:
                _exclude(row, reason)
            elif not task.get("instructions") or not criteria:
                _exclude(row, "Task instructions or rubric criteria missing")
            elif any(
                c.get("evaluation_options", {}).get("include_docx_redlines") for c in criteria
            ):
                _exclude(
                    row,
                    "Criterion requires tracked-change extraction; this adapter extracts accepted document text only",
                )
            elif any(
                Path(name).suffix.lower() not in TEXT_ARTIFACTS
                for c in criteria
                for name in c.get("deliverables", [])
            ):
                _exclude(row, "Criterion references an unsupported output file format")
            records.append(row)
    elif benchmark == "healthbench_professional":
        for task in _health_rows(source):
            # Prompt-equivalent records must share a group even if IDs differ.
            group = "conversation/" + _hash(task["conversation"])
            row = _base(benchmark, task["id"], [task, provenance], group, provenance)
            row.update(
                specialty=task.get("specialty"),
                difficulty=task.get("difficulty"),
                use_case=task.get("use_case"),
                original_split="eval",
            )
            if not task["conversation"] or task["conversation"][-1]["role"] != "user":
                _exclude(row, "Conversation does not end with a user message")
            elif any(
                not isinstance(message.get("content"), str) for message in task["conversation"]
            ):
                _exclude(row, "Non-text conversation content requires another transport")
            elif not any(r.get("points", 0) > 0 for r in task["rubric_items"]):
                _exclude(row, "No positive rubric denominator")
            records.append(row)
    elif benchmark == "putnambench":
        for path in sorted((source / "lean4/src").glob("*.lean")):
            text, task_id = path.read_text(), path.stem
            match = re.fullmatch(r"putnam_(\d{4})_[ab][1-6]", task_id)
            group = "competition/" + (match[1] if match else task_id)
            row = _base(benchmark, task_id, [text, provenance], group, provenance)
            theorem = re.search(r"\btheorem\s+" + re.escape(task_id) + r"\b", text)
            if text.count("sorry") != 1 or theorem is None or text.index("sorry") < theorem.end():
                _exclude(
                    row,
                    "Restricted proof-term protocol requires exactly one hole in the named theorem; answer-definition/multiple-hole tasks need another adapter",
                )
            records.append(row)
    elif benchmark in {"oragentbench", "terminal_bench_2"}:
        root = source / "harbor_tasks" if benchmark == "oragentbench" else source
        for path in sorted(root.glob("*/task.toml")):
            config, task = tomllib.loads(path.read_text()), path.parent
            task_id = task.name
            # Some OR versions encode instance variants as trailing numeric suffixes.
            group = re.sub(r"(?:_(?:instance|variant|seed))?[_-]\d+$", "", task_id)
            instruction = task / "instruction.md"
            content = [
                config,
                instruction.read_text() if instruction.is_file() else None,
                provenance,
            ]
            row = _base(benchmark, task_id, content, group, provenance)
            row["runtime_spec"] = config.get("environment", {})
            if not instruction.is_file():
                _exclude(row, "Task instruction.md missing from the downloaded release")
            elif benchmark == "terminal_bench_2":
                if task_id != "regex-log":
                    _exclude(
                        row,
                        "Task-specific native image, initial filesystem and clean artifact transfer not yet integrated; regex fixture cannot stand in for this task",
                    )
            elif not all(
                (task / name).exists()
                for name in ("environment/app", "solution/solve.sh", "tests/test.sh")
            ):
                _exclude(row, "Missing public app, native oracle or native test entrypoint")
            elif config.get("metadata", {}).get("requires_gurobi"):
                _exclude(row, "Requires licensed solver absent from the pinned local runtime")
            records.append(row)
    elif benchmark == "tau3_bench":
        tasks, splits = _tau_rows(source)
        for task in tasks:
            task_id = "retail/" + str(task["id"])
            criteria = task.get("evaluation_criteria") or {}
            actions = criteria.get("actions") or []
            user_ids = sorted(
                {
                    str(a.get("arguments", {}).get("user_id"))
                    for a in actions
                    if a.get("arguments", {}).get("user_id") is not None
                }
            )
            # Related customer-state tasks stay together, otherwise group identical scenario text.
            group = (
                "retail/customer/" + "/".join(user_ids)
                if user_ids
                else "retail/scenario/" + _hash(task["user_scenario"])
            )
            row = _base(benchmark, task_id, [task, provenance], group, provenance)
            row["native_task_id"] = str(task["id"])
            row["original_splits"] = [name for name, ids in splits.items() if task["id"] in ids]
            if not actions:
                _exclude(row, "No gold actions for positive/native end-state controls")
            elif criteria.get("nl_assertions"):
                _exclude(
                    row,
                    "Native NL assertion judge needs separate strict Terra routing; current generalized adapter admits deterministic-reward tasks only",
                )
            elif not (source / ".venv/bin/python").is_file():
                _exclude(row, "Pinned tau Python environment missing")
            records.append(row)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError(f"Duplicate native task ids in {benchmark}")
    return records


def inventory(root: Path | None = None) -> list[dict]:
    """Return every available task, including explicit unsupported-task exclusions."""
    return [row for benchmark in DOMAINS for row in _inventory_one(benchmark, root)]


def _load(benchmark: str, task_id: str, root: Path | None = None) -> tuple[Path, Any, Path | None]:
    source, _ = _source(benchmark, root)
    if benchmark == "gdpval":
        matches = [row for row in _gdp_rows(source) if row["task_id"] == task_id]
    elif benchmark == "healthbench_professional":
        matches = [row for row in _health_rows(source) if row["id"] == task_id]
    elif benchmark == "tau3_bench":
        matches = [row for row in _tau_rows(source)[0] if "retail/" + str(row["id"]) == task_id]
    else:
        matches = []
        if benchmark == "harvey_lab":
            task = _safe(source / "tasks", task_id)
            return source, json.loads((task / "task.json").read_text()), task
        if benchmark == "putnambench":
            task = _safe(source / "lean4/src", task_id + ".lean")
            return source, task.read_text(), task
        if benchmark in {"oragentbench", "terminal_bench_2"}:
            root = source / "harbor_tasks" if benchmark == "oragentbench" else source
            task = _safe(root, task_id)
            return source, tomllib.loads((task / "task.toml").read_text()), task
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one native task for {benchmark}/{task_id}; found {len(matches)}"
        )
    return source, matches[0], None


def preflight(
    benchmark: str,
    task_id: str,
    output: Path | None = None,
    image: str = IMAGE,
    *,
    native_controls: bool = False,
    root: Path | None = None,
) -> dict:
    """Resolve exactly one task and validate task schema/assets without inference."""
    if benchmark not in DOMAINS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    matches = [row for row in _inventory_one(benchmark, root) if row["id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicated exact task id: {benchmark}/{task_id}")
    record = matches[0]
    if not record["adapter_ready"]:
        return {
            **record,
            "ready": False,
            "runtime_validated": False,
            "structural_preflight_passed": False,
            "native_controls_passed": None,
        }
    source, task, path = _load(benchmark, task_id, root)
    assets: list[Path] = []
    try:
        if benchmark == "gdpval":
            criteria = json.loads(task["rubric_json"])
            if not criteria or not all(
                type(c.get("score")) in {int, float} and math.isfinite(c["score"]) for c in criteria
            ):
                raise ValueError("GDPval rubric scores malformed")
            if sum(max(0, c["score"]) for c in criteria) <= 0:
                raise ValueError("GDPval positive rubric denominator missing")
            assets = [
                _safe(source / "reference_files", str(Path(name).relative_to("reference_files")))
                for name in task["reference_files"]
            ]
        elif benchmark == "harvey_lab":
            assert path is not None
            docs = _legal_documents(source, path, task)
            assets = [p for p in docs.rglob("*") if p.is_file()]
            ids = [c["id"] for c in task["criteria"]]
            if len(ids) != len(set(ids)):
                raise ValueError("LAB criterion IDs are duplicated")
            for criterion in task["criteria"]:
                if not all(
                    isinstance(criterion.get(key), str) and criterion[key]
                    for key in ("id", "title", "match_criteria")
                ):
                    raise ValueError("Malformed LAB criterion")
                for name in criterion.get("deliverables", []):
                    _safe(Path("/placeholder"), name, exists=False)
        elif benchmark in {"oragentbench", "terminal_bench_2"}:
            assert path is not None
            assets = [p for p in path.rglob("*") if p.is_file()]
        elif benchmark == "putnambench":
            assert path is not None
            assets = [path]
        for asset in assets:
            if asset.is_symlink() or not asset.resolve().is_relative_to(source.resolve()):
                raise ValueError("Input asset escaped its release or is a symlink")
        record["input_assets_sha256"] = _hash(
            {str(p.relative_to(source)): _file_hash(p) for p in sorted(assets)}
        )
        record["input_file_count"] = len(assets)
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        _exclude(record, str(exc))
    report = {
        **record,
        "ready": record["adapter_ready"],
        "runtime_validated": False,
        "structural_preflight_passed": record["adapter_ready"],
        "native_controls_passed": None,
        "requested_runtime_image": image,
    }
    if native_controls:
        raise UnsupportedTask(
            "Standalone native-control preflight is not implemented for this adapter; evaluate runs formal/OR/terminal task controls before solving, rubric transport controls run separately"
        )
    if output is not None:
        write_json(Path(output) / "preflight.json", report)
    return report


def _policy(policy: dict) -> None:
    if (
        policy.get("model") != "gpt-6-astra"
        or policy.get("authentication") != "codex_chatgpt_account"
    ):
        raise ValueError("Extended adapters require the signed-in account Astra policy")
    if policy.get("reasoning_effort") != "xhigh":
        raise ValueError("Extended adapter solver reasoning is frozen at xhigh")
    options = judge_options(policy)
    if options != {"model": "gpt-5.6-terra", "effort": "medium"}:
        raise ValueError("Extended rubric judges are frozen at Terra/medium")
    if not isinstance(policy.get("instructions"), str) or not policy["instructions"].strip():
        raise ValueError("Missing shared harness instructions")


def _check_extraction(files: list[dict]) -> list[dict]:
    """Unreadable candidate files are submission failures, not missing evaluations.

    A failed Docker/extractor process raises in ``extract`` before this function.
    Known missing-runtime/permission errors also remain infrastructure failures.
    The frozen multitask protocol assigns zero to a submission containing any
    unreadable supported artifact; these cases stay in the task denominator.
    """
    failed = [
        {"filename": row["filename"], "parse_error": row["parse_error"]}
        for row in files
        if row.get("parse_error")
    ]
    for row in failed:
        message = str(row["parse_error"])
        if any(
            marker in message
            for marker in (
                "No module named",
                "Permission denied",
                "No such file or directory: 'pandoc'",
            )
        ):
            raise RuntimeError(f"Artifact extractor runtime failed: {message}")
    return failed


def _gdp(policy: dict, output: Path, image: str, task_id: str):
    from moevo.codex.document_tasks import IMAGE, extract, gdp_grade

    source, row, _ = _load("gdpval", task_id)
    workspace = output / "workspace"
    refs = workspace / "reference_files"
    refs.mkdir(parents=True)
    for name in row["reference_files"]:
        origin = _safe(source, name)
        shutil.copyfile(origin, refs / origin.name)
    (workspace / "output").mkdir()
    (workspace / "output").chmod(0o777)
    response = solve(
        policy,
        row["prompt"]
        + "\nInput files are in /workspace/reference_files. Save final deliverables in /workspace/output.",
        output,
        IMAGE,
    )
    files = extract(workspace / "output", output / "extracted.json")
    invalid = _check_extraction(files)
    if invalid:
        grade = {
            "score": 0.0,
            "invalid_submission": True,
            "invalid_artifacts": invalid,
            "rule": "Any unreadable supported candidate artifact makes the submission invalid; retain task in denominator",
            "criteria_not_judged": True,
        }
        write_json(output / "grade" / "grade.json", grade)
    else:
        grade = gdp_grade(
            row["prompt"], files, json.loads(row["rubric_json"]), output / "grade", judge=policy
        )
    return result(
        response,
        grade["score"],
        grade,
        task_id,
        "Public GDPval weighted rubric, actual artifact extraction, account Terra judge; not expert pairwise win rate",
        **judge_metadata(policy),
        artifact_count=len(files),
        visual_layout_evaluated=False,
        runtime_image=IMAGE,
        invalid_submission_rule="Unreadable supported candidate artifact scores zero, including its task in the denominator",
    )


def _legal(policy: dict, output: Path, image: str, task_id: str):
    from moevo.codex.document_tasks import IMAGE, extract, legal_verdict

    source, task, path = _load("harvey_lab", task_id)
    assert path is not None
    workspace = output / "workspace"
    shutil.copytree(_legal_documents(source, path, task), workspace / "documents")
    (workspace / "output").mkdir()
    (workspace / "output").chmod(0o777)
    response = solve(
        policy,
        task["instructions"]
        + "\nRead /workspace/documents. Save all requested deliverables in /workspace/output, preserving requested filenames.",
        output,
        IMAGE,
    )
    extracted = extract(workspace / "output", output / "extracted.json")
    invalid = _check_extraction(extracted)
    files = {row["filename"]: row for row in extracted}
    if invalid:
        grade = {
            "all_pass": False,
            "n_passed": 0,
            "n_criteria": len(task["criteria"]),
            "criteria_results": [],
            "invalid_submission": True,
            "invalid_artifacts": invalid,
            "criteria_not_judged": True,
            "rule": "Any unreadable supported candidate artifact makes the submission invalid; retain task in denominator",
        }
        write_json(output / "grade.json", grade)
        return result(
            response,
            0.0,
            grade,
            task_id,
            "LAB account criterion protocol with declared invalid-artifact task failure rule",
            **judge_metadata(policy),
            criterion_pass_rate=0.0,
            artifact_count=len(files),
            runtime_image=IMAGE,
        )

    def grade_one(item):
        index, criterion = item
        names = criterion.get("deliverables") or list(files)
        text = "\n\n".join(
            f"## {name}\n" + json.dumps(files.get(name, {"missing_file": name}), ensure_ascii=False)
            for name in names
        )
        verdict = legal_verdict(
            task["title"],
            text or "(No output files)",
            criterion["title"],
            criterion["match_criteria"],
            output / "grade" / f"criterion-{index:03d}",
            judge=policy,
        )
        return {"id": criterion["id"], **verdict}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        grades = list(pool.map(grade_one, enumerate(task["criteria"])))
    passed = sum(grade["verdict"] == "pass" for grade in grades)
    grade = {
        "all_pass": passed == len(grades),
        "n_passed": passed,
        "n_criteria": len(grades),
        "criteria_results": grades,
    }
    write_json(output / "grade.json", grade)
    return result(
        response,
        float(grade["all_pass"]),
        grade,
        task_id,
        "Official LAB criterion prompt and all-pass rule; exact filename matching and accepted-text extraction; account Terra judge variant",
        **judge_metadata(policy),
        criterion_pass_rate=passed / len(grades),
        artifact_count=len(files),
        runtime_image=IMAGE,
    )


def _health(policy: dict, output: Path, image: str, task_id: str):
    from moevo.codex.health_pilot import grade_response

    _, row, _ = _load("healthbench_professional", task_id)
    prompt = policy["instructions"] + "\nRespond to the final user message:\n\n"
    prompt += "\n\n".join(
        f"{message['role']}: {message['content']}" for message in row["conversation"]
    )
    with tempfile.TemporaryDirectory(prefix="moevo-health-multitask-") as cwd:
        response = run_codex(
            prompt,
            cwd=Path(cwd),
            model=policy["model"],
            effort=policy["reasoning_effort"],
            tools=False,
            timeout=policy["timeout_seconds"],
            log_dir=output / "agent",
        )
    grade = grade_response(row, response.text, output / "grade", judge=policy)
    return result(
        response,
        grade["raw_rubric_score"],
        grade,
        task_id,
        "Public HealthBench weighted scorer; account Astra solver/Terra judge; text-only external protocol variant",
        **judge_metadata(policy),
        strict_pass=grade["all_criteria_passed"],
    )


def _putnam(policy: dict, output: Path, image: str, task_id: str):
    from moevo.codex.formal_tasks import IMAGE, compile_lean, validate_proof_term

    _, original, _ = _load("putnambench", task_id)
    image_id = json.loads(
        subprocess.check_output(["docker", "image", "inspect", IMAGE], text=True, timeout=30)
    )[0]["Id"]
    controls = {
        "positive": compile_lean(
            "import Mathlib\ntheorem smoke : (1 : Nat) + 1 = 2 := by norm_num",
            "smoke",
            output / "positive_control",
        ),
        "negative": compile_lean(
            "import Mathlib\ntheorem smoke : (1 : Nat) + 1 = 3 := by norm_num",
            "smoke",
            output / "negative_control",
        ),
        "admission": compile_lean(
            "import Mathlib\ntheorem smoke : False := by sorry",
            "smoke",
            output / "admission_control",
        ),
    }
    if [controls[name]["score"] for name in ("positive", "negative", "admission")] != [1, 0, 0]:
        raise RuntimeError("Lean compiler/admission controls failed")
    workspace = output / "workspace"
    workspace.mkdir()
    (workspace / "task.lean").write_text(original)
    response = solve(
        policy,
        "Prove the named theorem in /workspace/task.lean using the pinned Lean/Mathlib at /opt/putnam. "
        "Check with cd /opt/putnam && lake env lean /workspace/answer.lean. Save ONLY the proof term "
        "replacing sorry in /workspace/proof.txt. Do not change the statement; no admissions, "
        "new declarations, custom axioms, native_decide or compiler commands are allowed.",
        output,
        IMAGE,
    )
    artifact = _safe(workspace, "proof.txt", exists=False)
    proof = artifact.read_text() if artifact.is_file() else ""
    try:
        validate_proof_term(proof)
    except ValueError as exc:
        grade = {"score": 0.0, "kernel_checked": False, "invalid_submission": str(exc)}
    else:
        grade = compile_lean(original.replace("sorry", f"({proof})"), task_id, output / "grade")
    return result(
        response,
        grade["score"],
        grade,
        task_id,
        "Pinned PutnamBench theorem and Lean/Mathlib kernel; strict single-proof-term and local resource protocol",
        controls=controls,
        task_sha256=hashlib.sha256(original.encode()).hexdigest(),
        agent_image_id=image_id,
        verifier_image_id=image_id,
    )


def _or_scalar_reward(scored: dict, grade_directory: Path) -> float:
    """Read the native reward, including its explicit missing-submission branch.

    Upstream tests/test.sh writes reward.txt=0 and feasibility=quality=0 for a
    missing solution, without scalar_reward. Other branches write the explicit
    scalar (feasibility + quality)/3. No arbitrary absent field becomes zero.
    """
    evaluation, details = scored["evaluation"], scored["reward"]
    if type(evaluation.get("feasible")) is not bool:
        raise ValueError("Native OR evaluator omitted its boolean feasibility result")
    if any(
        "Could not parse evaluator JSON" in str(error) for error in evaluation.get("errors", [])
    ):
        raise RuntimeError("Native OR grader failed to produce evaluator JSON")
    feasibility, quality = details.get("feasibility"), details.get("quality")
    if (
        type(feasibility) not in {int, float}
        or feasibility not in {0.0, 1.0}
        or type(quality) not in {int, float}
        or not math.isfinite(quality)
        or not 0 <= quality <= 2
        or bool(feasibility) != evaluation["feasible"]
    ):
        raise ValueError("Native OR reward dimensions are missing or inconsistent")
    native = float((grade_directory / "reward.txt").read_text().strip())
    if not math.isfinite(native) or not 0 <= native <= 1:
        raise ValueError("Native OR reward.txt is not a finite normalized score")
    if "scalar_reward" not in details:
        if not (
            details.get("quality_status") == "missing_solution"
            and evaluation["feasible"] is False
            and feasibility == 0
            and quality == 0
            and native == 0
        ):
            raise ValueError(
                "Missing native OR scalar outside the documented missing_solution branch"
            )
        return native
    scalar = details["scalar_reward"]
    if (
        type(scalar) not in {int, float}
        or not math.isfinite(scalar)
        or not math.isclose(scalar, native, rel_tol=1e-12, abs_tol=1e-12)
        or not math.isclose(scalar, (feasibility + quality) / 3.0, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError("Native OR scalar, reward.txt and documented formula disagree")
    return native


def _operations(policy: dict, output: Path, image: str, task_id: str):
    from experiments.validate_oragentbench import grade

    _, _, task = _load("oragentbench", task_id)
    assert task is not None
    public = task / "environment/app"
    oracle = output / "oracle"
    shutil.copytree(public, oracle)
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
            "--pids-limit=256",
            "--tmpfs",
            "/tmp:rw,size=512m",
            "--mount",
            f"type=bind,source={oracle},target=/app",
            "--mount",
            f"type=bind,source={task / 'solution'},target=/solution,readonly",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            image,
            "timeout",
            "--kill-after=5",
            "330",
            "bash",
            "/solution/solve.sh",
        ],
        capture_output=True,
        text=True,
        timeout=360,
        check=True,
    )
    (output / "oracle_stdout.txt").write_text(completed.stdout)
    positive = grade(image, task, oracle, output / "positive_control")
    negative_workspace = output / "negative"
    shutil.copytree(public, negative_workspace)
    negative = grade(image, task, negative_workspace, output / "negative_control")
    if not positive["evaluation"]["feasible"] or negative["evaluation"]["feasible"]:
        raise ValueError("OR native oracle/no-op controls failed")
    workspace = output / "workspace"
    shutil.copytree(public, workspace)
    (workspace / "submissions").mkdir(exist_ok=True)
    (workspace / "submissions").chmod(0o777)
    response = solve(
        policy,
        (task / "instruction.md").read_text()
        + "\n/app and /workspace contain the same public task files.",
        output,
        image,
    )
    pristine = output / "evaluation"
    shutil.copytree(public, pristine)
    for path in (workspace / "submissions").rglob("*"):
        if path.is_symlink():
            raise ValueError("OR submission contains a symlink")
        if path.is_file():
            target = _safe(
                pristine / "submissions",
                path.relative_to(workspace / "submissions").as_posix(),
                exists=False,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    scored = grade(image, task, pristine, output / "grade")
    score = _or_scalar_reward(scored, output / "grade")
    return result(
        response,
        score,
        scored,
        task_id,
        "Official OR oracle/grader and pristine-input artifact validation; local2CPU/4GB offline runtime variant",
        from_scratch_replay_validated=False,
        controls={"positive": positive, "negative": negative},
        scalar_reward_source="Validated native reward.txt; missing_solution is a scored zero",
    )


def _terminal(policy: dict, output: Path, image: str, task_id: str):
    if task_id != "regex-log":
        raise UnsupportedTask("Native task-specific Terminal-Bench runtime not integrated")
    from moevo.codex.terminal_tasks import regex

    return regex(policy, output, image)


def _tau(policy: dict, output: Path, image: str, task_id: str):
    source, _, _ = _load("tau3_bench", task_id)
    policy_path = output / "policy.json"
    write_json(policy_path, policy)
    with (output / "stdout.txt").open("w") as stdout, (output / "stderr.txt").open("w") as stderr:
        process = subprocess.run(
            [
                str(source / ".venv/bin/python"),
                "-m",
                "moevo.codex.multitask_extended",
                "--tau-worker",
                task_id,
                "--output",
                str(output),
                "--policy",
                str(policy_path),
            ],
            cwd=ROOT,
            env=account_environment(),
            stdout=stdout,
            stderr=stderr,
            timeout=policy["timeout_seconds"] + 180,
        )
    if process.returncode:
        raise RuntimeError(
            "Tau exact-task worker failed: " + (output / "stderr.txt").read_text()[-1800:]
        )
    return json.loads((output / "native-report.json").read_text())


HANDLERS = {
    "gdpval": _gdp,
    "harvey_lab": _legal,
    "healthbench_professional": _health,
    "putnambench": _putnam,
    "oragentbench": _operations,
    "terminal_bench_2": _terminal,
    "tau3_bench": _tau,
}


def evaluate(benchmark: str, task_id: str, policy: dict, output: Path, image: str) -> dict:
    """Execute precisely the requested task; unsupported or failed tasks raise."""
    _policy(policy)
    metadata = preflight(benchmark, task_id)
    if not metadata["ready"]:
        raise UnsupportedTask(metadata["exclusion_reason"])
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Use an empty output attempt directory; existing evidence is immutable"
        )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preflight.json", metadata)
    report = HANDLERS[benchmark](policy, output, image, task_id)
    if report.get("task_id") != task_id:
        raise ValueError("Adapter returned a different task id than requested")
    if not report.get("e2e_validated"):
        raise ValueError("Native execution/grading did not complete")
    if type(report.get("score")) not in {int, float} or not math.isfinite(report["score"]):
        raise ValueError("Native score is not finite")
    report.update(
        benchmark=benchmark,
        content_sha256=metadata["content_sha256"],
        input_assets_sha256=metadata["input_assets_sha256"],
        exposed=metadata["exposed"],
        original_score_retained=True,
        runtime_controls_scope="Protocol-specific; transport rubric controls should be run once per frozen judge configuration",
    )
    write_json(output / "report.json", report)
    return report


def _tau_worker(task_id: str, output: Path, policy: dict) -> None:
    # Runs only inside upstream Python3.12; no persistent global monkeypatch in callers.
    _policy(policy)
    clean = account_environment()
    os.environ.clear()
    os.environ.update(clean)
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["LOG_LEVEL"] = "ERROR"
    from loguru import logger
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.run import get_tasks, run_single_task

    from moevo.codex.tau_pilot import AccountTransport, install_transport, validate_controls

    logger.remove()
    native_id = task_id.removeprefix("retail/")
    tasks = [
        task for task in get_tasks("retail", task_split_name="base") if str(task.id) == native_id
    ]
    if len(tasks) != 1:
        raise ValueError("Tau exact native task id not found")
    task = tasks[0]
    if task.evaluation_criteria is None or task.evaluation_criteria.nl_assertions:
        raise UnsupportedTask("This worker supports native deterministic-reward retail tasks only")
    validate_controls(task, "retail", output)

    class TaskTransport(AccountTransport):
        def generate(self, *args, call_name=None, **kwargs):
            if call_name not in {"agent_response", "user_simulator_response"}:
                raise UnsupportedTask(f"Unexpected judge/model call role: {call_name}")
            return super().generate(*args, call_name=call_name, **kwargs)

    transport = TaskTransport(output / "calls", policy["timeout_seconds"], policy["instructions"])
    routes = install_transport(transport)
    config = TextRunConfig(
        domain="retail",
        agent="llm_agent",
        user="user_simulator",
        llm_agent="gpt-6-astra",
        llm_user="gpt-6-astra",
        llm_args_agent={},
        llm_args_user={},
        max_steps=50,
        timeout=policy["timeout_seconds"],
        auto_review=False,
        review_model="gpt-6-astra",
    )
    simulation = run_single_task(
        config,
        task,
        seed=policy["seed"],
        evaluation_type=EvaluationType.ALL,
        save_dir=output,
        auto_review=False,
    )
    write_json(output / "simulation.json", simulation.model_dump(mode="json"))
    if not transport.calls or simulation.reward_info is None:
        raise ValueError("Tau simulation produced no model calls or reward")
    reward = simulation.reward_info.model_dump(mode="json")
    usage: dict[str, int] = defaultdict(int)
    for call in transport.calls:
        for key, value in call["usage"].items():
            usage[key] += value
    report = {
        "task_id": task_id,
        "native_task_id": native_id,
        "score": reward["reward"],
        "grade": reward,
        "e2e_validated": True,
        "termination": simulation.termination_reason.value,
        "model_calls": len(transport.calls),
        "usage": dict(usage),
        "account_routes": routes,
        "protocol": "Native retail simulator, deterministic reward, account Astra agent/user, JSON tool transport,50-step local cap; NL-judge tasks excluded",
    }
    write_json(output / "native-report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-worker", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    _tau_worker(args.tau_worker, args.output, json.loads(args.policy.read_text()))
