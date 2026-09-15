"""Preserve observable live candidate source without altering an evaluator or scores.

The existing scheduler uses temporary files and saves only admitted programs.
This observer accepts a temporary source only when its hash matches an actual
task-ledger candidate. It cannot recover a discarded file that already vanished.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from moevo.reporting.source import code_hash, instruction_literal


def capture(root: Path, temporary: Path) -> int:
    ledger = root / "evaluation_state.json"
    if not ledger.exists():
        return 0
    state = json.loads(ledger.read_text())
    known = {r["candidate_sha256"] for r in state.get("cache", {}).values()}
    copied = 0
    for source in temporary.glob("moevo-tasks-*/candidate.py"):
        try:
            code = source.read_text()
            sha = code_hash(code)
            if sha not in known:
                continue
            instruction_literal(code)
        except (FileNotFoundError, SyntaxError, ValueError):
            continue
        destination = root / "source_observations" / f"{sha}.py"
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            destination.write_text(code)
            copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        active = False
        for root in args.run:
            count = capture(root, Path(tempfile.gettempdir()))
            if count:
                print(f"{root.name}: preserved {count} observed candidate source files", flush=True)
            run = root / "run.json"
            active |= run.exists() and json.loads(run.read_text()).get("status") == "running"
        if args.once or not active:
            return
        time.sleep(3)


if __name__ == "__main__":
    main()
