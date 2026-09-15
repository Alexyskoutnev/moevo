"""One real account-Astra task per benchmark, with explicit validation status."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

from experiments.run_submission import CATALOG, load_submission
from moevo.codex.client import require_chatgpt_login
from moevo.codex.domain_tasks import HANDLERS
from moevo.codex.finance_pilot import write_json
from moevo.codex.judging import judge_metadata

ROOT = Path(__file__).resolve().parents[1]


def handler_for(name):
    if name in HANDLERS:
        return HANDLERS[name]
    from moevo.codex.extended_tasks import HANDLERS as EXTENDED_HANDLERS

    if name not in EXTENDED_HANDLERS:
        raise RuntimeError(f"No runnable adapter available for {name}")
    return EXTENDED_HANDLERS[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "submissions/superharness-astra-terra-pilot/submission.json",
    )
    parser.add_argument("--benchmarks", nargs="+", choices=list(CATALOG), default=list(CATALOG))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/mini-suite/astra-terra-pilot"
    )
    parser.add_argument("--concurrency", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    policy = load_submission(args.submission)
    if not set(args.benchmarks).issubset(policy["benchmarks"]):
        parser.error("Selected benchmarks are absent from the policy manifest")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    snapshot = output / "submission.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != policy:
        parser.error("Policy changed; use a new output directory")
    write_json(snapshot, policy)
    version = require_chatgpt_login()
    image = json.loads(
        subprocess.check_output(
            ["docker", "image", "inspect", "docker.io/library/moevo-or-runtime:20260915"],
            text=True,
            timeout=30,
        )
    )[0]["Id"]
    lock = threading.Lock()
    rows = {
        name: {
            "benchmark": name,
            "domain": domain,
            "name": title,
            "status": "not_run",
            "score": None,
            "e2e_validated": False,
        }
        for name, (domain, title) in CATALOG.items()
        if name in policy["benchmarks"]
    }
    for name in rows:
        saved = output / name / "report.json"
        if saved.exists():
            record = json.loads(saved.read_text())
            if record.get("policy_sha256") != signature:
                raise ValueError("Cached benchmark used a different policy")
            rows[name] = record

    def save():
        report = {
            "purpose": "One real development task per benchmark; integration validation only",
            "model": policy["model"],
            "effort": policy["reasoning_effort"],
            **judge_metadata(policy),
            "authentication": "Codex ChatGPT account",
            "cli_version": version,
            "policy_sha256": signature,
            "updated_utc": datetime.now(UTC).isoformat(),
            "dummy": False,
            "evolved": False,
            "full_benchmark_scores": False,
            "validated": sum(r["e2e_validated"] for r in rows.values()),
            "real_tasks_scored": sum(
                r["status"] in {"scored", "needs_audit"} for r in rows.values()
            ),
            "results": list(rows.values()),
        }
        write_json(output / "summary.json", report)
        lines = [
            "# Astra mini-suite validation",
            "",
            report["purpose"],
            "",
            f"Authentication: {report['authentication']}; model: {policy['model']} / {policy['reasoning_effort']}.",
            "",
            "| Benchmark | Status | Native score | End to end |",
            "| --- | --- | ---: | --- |",
        ]
        for r in rows.values():
            score = "—" if r["score"] is None else str(round(r["score"], 4))
            lines.append(
                f"| {r['name']} | {r['status']} | {score} | {'yes' if r['e2e_validated'] else 'pending'} |"
            )
        lines += [
            "",
            "Each score covers one task. Protocol variants and errors are in summary.json.",
            "",
        ]
        (output / "STATUS.md").write_text("\n".join(lines))

    def run(name):
        with lock:
            if rows[name]["status"] in {"scored", "needs_audit"}:
                return
            rows[name].update(status="running", score=None)
            save()
        directory = output / name
        attempt = directory / f"attempt-{len(list(directory.glob('attempt-*'))) + 1:02d}"
        attempt.mkdir(parents=True)
        record = {
            **rows[name],
            "policy_sha256": signature,
            "attempt": str(attempt.relative_to(output)),
        }
        print(json.dumps({"benchmark": name, "status": "running"}), flush=True)
        try:
            handler = handler_for(name)
            code = Path(handler.__code__.co_filename)
            record["adapter_sha256"] = hashlib.sha256(code.read_bytes()).hexdigest()
            record["runtime_image_id"] = image
            evaluated = handler(policy, attempt, image)
            record.update(evaluated)
            record["status"] = "scored" if record["e2e_validated"] else "needs_audit"
        except Exception as exc:
            record.update(status="blocked", score=None, e2e_validated=False, error=str(exc))
            write_json(attempt / "error.json", {"type": type(exc).__name__, "error": str(exc)})
        write_json(directory / "report.json", record)
        with lock:
            rows[name] = record
            save()
        print(
            json.dumps({k: record.get(k) for k in ("benchmark", "status", "score", "error")}),
            flush=True,
        )

    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(run, args.benchmarks))
    save()


if __name__ == "__main__":
    main()
