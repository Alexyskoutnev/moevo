"""Account-Astra E2E check for BizFinBench.v2 financial computation."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import sys
from pathlib import Path

from moevo.codex.client import require_chatgpt_login
from moevo.codex.container_runtime import solve_in_container
from moevo.codex.finance_pilot import write_json
from moevo.codex.headroom import assess_headroom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.count <= 128:
        raise ValueError("Count must be 1–128 for this development screen")
    source = Path("data/external/bizfinbench2").resolve()
    sys.path.insert(0, str(source))
    scorer = source / "benchmark_code/BizFinBench.v2/eval_financial_quantitative_computation.py"
    spec = importlib.util.spec_from_file_location("bizfin_numeric", scorer)
    assert spec is not None and spec.loader is not None
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    raw = Path("data/raw/bizfinbench2/en/financial_quantitative_computation_en.jsonl")
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    indices = random.Random(args.seed).sample(range(len(rows)), 128)[: args.count]
    root = Path("results/domain_validation/bizfinbench2_numeric").resolve()
    root.mkdir(parents=True, exist_ok=True)
    version = require_chatgpt_login()
    scores = []
    for index in indices:
        row = rows[index]
        output = root / f"task-{index:04d}"
        output.mkdir(parents=True, exist_ok=True)
        if (output / "report.json").exists():
            report = json.loads((output / "report.json").read_text())
            if report["timeout_seconds"] != args.timeout:
                raise ValueError("Cached result used a different inference budget")
            scores.append(report["score"])
            continue

        def grade(prediction: str, name: str, row=row, output=output) -> dict:
            record = copy.deepcopy(row)
            record["predict_result"] = prediction
            path = output / (name + ".jsonl")
            path.write_text(json.dumps(record) + "\n")
            official.evaluation(str(path))
            result = json.loads(path.read_text())
            if "error" in result["eval_result"]:
                raise ValueError("Official grader exception: " + result["eval_result"]["error"])
            return result

        gold = row["choices"][0]["message"]["content"][0]["text"]
        positive = grade(json.dumps({"answer": gold}), "positive_control")
        negative = grade(json.dumps({"answer": "no numerical answer"}), "negative_control")
        if positive["score"] != 1 or negative["score"] != 0:
            raise ValueError("Official numeric grader failed controls")
        prompt_parts = []
        for message in row["messages"]:
            if message["role"] == "assistant":
                continue
            content = message["content"]
            if isinstance(content, list):
                content = "\n".join(part["text"] for part in content if part.get("type") == "text")
            prompt_parts.append(message["role"].upper() + ": " + content)
        response = solve_in_container(
            "\n\n".join(prompt_parts)
            + "\n\nUse the benchmark run tool for calculations. Follow the requested JSON answer format. "
            "Write numerical answers in decimal notation rather than scientific notation.",
            output / "workspace",
            output / "agent",
            timeout=args.timeout,
        )
        result = grade(response.text, "prediction")
        report = {
            "dataset": "BizFinBench.v2 English financial_quantitative_computation",
            "row_index": index,
            "seed": args.seed,
            "source_file": str(raw),
            "source": json.loads((source / "source_manifest.json").read_text()),
            "model": "gpt-6-astra",
            "effort": "xhigh",
            "cli_version": version,
            "timeout_seconds": args.timeout,
            "score": result["score"],
            "grade": result["eval_result"],
            "response": response.text,
            "usage": response.usage,
            "duration_s": response.duration_s,
            "e2e_smoke_passed": True,
            "semantic_failure_audited": False,
        }
        write_json(output / "report.json", report)
        scores.append(result["score"])
        print(
            json.dumps({"row_index": index, "score": result["score"], "tasks_done": len(scores)}),
            flush=True,
        )
    summary = {"indices": indices, **assess_headroom(scores)}
    write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
