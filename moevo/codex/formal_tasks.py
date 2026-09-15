"""One Putnam proof attempt checked by the release's pinned Lean kernel."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from moevo.codex.domain_tasks import ROOT, result, solve
from moevo.codex.finance_pilot import write_json

IMAGE = "docker.io/library/moevo-putnam-runtime:20260915"
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
# This adapter accepts a proof term, never new declarations or compiler commands.
FORBIDDEN = re.compile(
    r"\b(sorry|admit|axiom|unsafe|run_cmd|run_elab|elab|macro|syntax|initialize|"
    r"native_decide|implemented_by|extern|import|theorem|def|opaque|constant|"
    r"set_option|include|omit|namespace|section|end)\b|#"
)


def validate_proof_term(proof: str) -> None:
    if not proof.strip() or len(proof) > 100_000 or FORBIDDEN.search(proof):
        raise ValueError("Expected a nonempty proof term without admissions or declarations")


def accepted_axioms(stdout: str, theorem: str) -> bool:
    matches = re.findall(
        rf"^'{re.escape(theorem)}' (does not depend on any axioms|depends on axioms: \[[^\]]*\])$",
        stdout,
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        return False
    if matches[0] == "does not depend on any axioms":
        return True
    axioms = matches[0].split("[", 1)[1].removesuffix("]")
    return set(axioms.replace("\n", " ").replace(",", " ").split()) <= ALLOWED_AXIOMS


def compile_lean(source: str, theorem: str, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output / "Check.lean").write_text(source + f"\n#print axioms {theorem}\n")
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
            "--user=1000:1000",
            "--tmpfs",
            "/tmp:rw,size=512m",
            "--mount",
            f"type=bind,source={output.resolve()},target=/check,readonly",
            "--workdir=/opt/putnam",
            IMAGE,
            "timeout",
            "--kill-after=5",
            "180",
            "lake",
            "env",
            "lean",
            "/check/Check.lean",
        ],
        capture_output=True,
        text=True,
        timeout=200,
    )
    (output / "stdout.txt").write_text(completed.stdout)
    (output / "stderr.txt").write_text(completed.stderr)
    if completed.returncode not in {0, 1, 124, 137}:
        raise RuntimeError("Lean runtime failed: " + completed.stderr[-1000:])
    passed = completed.returncode == 0 and accepted_axioms(completed.stdout, theorem)
    grade = {
        "score": float(passed),
        "returncode": completed.returncode,
        "kernel_checked": passed,
        "allowed_axioms": sorted(ALLOWED_AXIOMS),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    write_json(output / "grade.json", grade)
    return grade


def putnam(policy: dict, output: Path, image: str):
    # Fixed before the first model call. No selection based on model outcomes.
    task_id = "putnam_2010_a4"
    task = ROOT / f"data/external/putnambench/lean4/src/{task_id}.lean"
    original = task.read_text()
    if original.count("sorry") != 1:
        raise ValueError("Expected exactly one proof hole in the pinned task")
    image_info = json.loads(
        subprocess.check_output(["docker", "image", "inspect", IMAGE], text=True, timeout=30)
    )[0]
    positive = compile_lean(
        "import Mathlib\ntheorem smoke : (1 : Nat) + 1 = 2 := by norm_num",
        "smoke",
        output / "positive_control",
    )
    negative = compile_lean(
        "import Mathlib\ntheorem smoke : (1 : Nat) + 1 = 3 := by norm_num",
        "smoke",
        output / "negative_control",
    )
    admitted = compile_lean(
        "import Mathlib\ntheorem smoke : False := by sorry", "smoke", output / "admission_control"
    )
    if positive["score"] != 1 or negative["score"] != 0 or admitted["score"] != 0:
        raise RuntimeError("Lean positive/negative/admitted-proof controls failed")
    workspace = output / "workspace"
    workspace.mkdir()
    (workspace / "task.lean").write_text(original)
    response = solve(
        policy,
        "Prove the theorem in /workspace/task.lean. Lean 4.27.0 and the pinned Mathlib are installed. "
        "Use `cd /opt/putnam && lake env lean /workspace/answer.lean` to check your work. "
        "Save ONLY the proof term replacing sorry in /workspace/proof.txt, usually beginning with by. "
        "Do not change the theorem statement. No admissions, new declarations, compiler commands, "
        "native_decide, or custom axioms are accepted. Standard kernel-checked tactics are permitted. "
        "The final check inserts your proof term into the original statement and checks its axioms.",
        output,
        IMAGE,
    )
    proof_file = workspace / "proof.txt"
    if proof_file.is_symlink() or (
        proof_file.exists() and not proof_file.resolve().is_relative_to(workspace.resolve())
    ):
        raise ValueError("Invalid proof artifact path")
    proof = proof_file.read_text() if proof_file.is_file() else ""
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
        "Pinned PutnamBench theorem, Lean 4.27.0 / Mathlib kernel; restricted proof-term protocol and local budget variant",
        controls={"positive": positive, "negative": negative, "admission": admitted},
        agent_image_id=image_info["Id"],
        verifier_image_id=image_info["Id"],
        task_sha256=hashlib.sha256(original.encode()).hexdigest(),
        control_scope="Compiler and axiom validation; no published oracle proof for this Putnam task",
    )
