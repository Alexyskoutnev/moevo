"""Versioned binding repair for account-judge audit hooks in frozen workers.

Importing the moevo package can preload judging.run_codex before the original
worker installs its client hook. Rebind that alias after the hook is installed.
Prompts, models, graders, worker leases and charge accounting stay unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments import run_reference_baselines as frozen

PROTOCOL = "reference-account-judge-binding-v2"


def install_binding_repair() -> None:
    original_install = frozen.install_worker_hooks

    def corrected_install(output, run, task_id, attempt_id):
        from moevo.codex import client, judging

        original_install(output, run, task_id, attempt_id)
        # The package initializer can import judging before client is hooked.
        judging.run_codex = client.run_codex
        row = frozen.read_json(output / "reference_state.json")["tasks"][task_id]
        if row["attempts"][-1]["id"] != attempt_id:
            raise ValueError("Transport repair attempt identity mismatch")
        destination = output / row["attempts"][-1]["output"]
        frozen.write_json(
            destination / "transport-amendment.json",
            {
                "protocol": PROTOCOL,
                "attempt_id": attempt_id,
                "source_sha256": frozen.file_hash(Path(__file__)),
                "rebound_alias": "moevo.codex.judging.run_codex",
                "binding_matches_audited_client": judging.run_codex is client.run_codex,
                "prompt_model_grading_changes": False,
            },
        )

    frozen.install_worker_hooks = corrected_install


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    run = frozen.read_json(output / "run.json")
    amendment = frozen.read_json(args.amendment)
    if (
        amendment["original_identity_sha256"] != run["identity_sha256"]
        or amendment["worker_protocol"] != PROTOCOL
        or amendment["worker_source_sha256"] != frozen.file_hash(Path(__file__))
    ):
        raise ValueError("Worker transport amendment identity mismatch")
    install_binding_repair()
    frozen.worker(output, args.task, args.attempt)


if __name__ == "__main__":
    main()
