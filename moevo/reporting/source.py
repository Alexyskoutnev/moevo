"""Read-only views of the exact instruction component used by an experiment."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path


def code_hash(code: str) -> str:
    return hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()


def instruction_literal(code: str) -> str:
    values = []
    for node in ast.parse(code).body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "INSTRUCTIONS"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            values.append(node.value.value)
        else:
            raise ValueError("Source is outside the frozen shared-instruction scope")
    if len(values) != 1 or not 1 <= len(values[0].strip()) <= 8000:
        raise ValueError("Invalid instruction component")
    return values[0]


def source_view(root: Path, database: dict, state: dict) -> dict:
    seed_path = root / "seed.py"
    if not seed_path.exists():
        return {"versions": [], "fixed_files": []}
    seed = seed_path.read_text()
    instruction_literal(seed)
    sources = {code_hash(seed): seed}
    complete = {}
    for program in database.get("all_programs", []):
        code = program["solution"]
        try:
            instruction_literal(code)
        except (SyntaxError, ValueError):
            continue
        sha = code_hash(code)
        sources[sha] = code
        complete[sha] = program
    known = {r["candidate_sha256"] for r in state.get("cache", {}).values()}
    for path in sorted((root / "source_observations").glob("*.py")):
        code = path.read_text()
        sha = code_hash(code)
        if path.stem != sha or sha not in known:
            continue
        try:
            instruction_literal(code)
        except (SyntaxError, ValueError):
            continue
        sources[sha] = code
    screen_events = {
        e["candidate_sha256"]: e
        for e in state.get("events", [])
        if e.get("stage") == "screen" and e.get("candidate_sha256")
    }
    versions = []
    for sha, code in sources.items():
        baseline = sha == code_hash(seed)
        event = screen_events.get(sha, {})
        program = complete.get(sha)
        phase = (
            "starting_agent"
            if baseline
            else "fully_tested"
            if program
            else "screened_out"
            if event and not (event.get("improved") or event.get("audit"))
            else "evaluation_in_progress"
        )
        versions.append(
            {
                "sha256": sha,
                "is_baseline": baseline,
                "status": phase,
                "screen": event.get("screen"),
                "code": code,
                "instructions": instruction_literal(code),
                "diff": "".join(
                    difflib.unified_diff(
                        seed.splitlines(keepends=True),
                        code.splitlines(keepends=True),
                        fromfile="starting_agent.py",
                        tofile="candidate.py",
                    )
                ),
                "characters": len(instruction_literal(code)),
                "tasks_scored": sum(
                    r["candidate_sha256"] == sha for r in state.get("cache", {}).values()
                ),
            }
        )
    # Fixed files are shown only if their bytes match the running protocol's identity.
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    repository = Path(__file__).resolve().parents[2]
    files = []
    for relative in [
        "experiments/sliced_task_adapter.py",
        "experiments/pilot_task_adapter.py",
        "moevo/codex/container_runtime.py",
        "moevo/codex/container_server.py",
    ]:
        expected = protocol.get("sources", {}).get(relative)
        path = repository / relative
        if (
            expected
            and path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == expected
        ):
            files.append({"path": relative, "code": path.read_text(), "sha256": expected})
    return {
        "versions": versions,
        "fixed_files": files,
        "scope": "Shared INSTRUCTIONS string; execution code, model, tools, limits and graders are fixed.",
    }
