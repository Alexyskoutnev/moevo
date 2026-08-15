#!/usr/bin/env python3
"""Run GDPval evaluation using Claude Code, Gemini CLI, or Codex.

Usage:
    # Quick test with Claude Code (default)
    python scripts/run_gdpval_eval.py --n 1

    # Run with Gemini CLI
    python scripts/run_gdpval_eval.py --agent gemini --n 1

    # Run with OpenAI Codex
    python scripts/run_gdpval_eval.py --agent codex --n 1

    # Run a specific dev slice
    python scripts/run_gdpval_eval.py --agent claude --slice S1

    # Run an eval slice
    python scripts/run_gdpval_eval.py --agent claude --slice E1

    # Full evaluation (all 220 tasks)
    python scripts/run_gdpval_eval.py --agent gemini --all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from typing import TYPE_CHECKING

from moevo.data.registry import DatasetRegistry
from moevo.data.sampler import UnifiedSampler
from moevo.eval.agents import AGENTS
from moevo.eval.runner import GDPvalRunner

if TYPE_CHECKING:
    from moevo.data.types import Sample


def load_zipper_slice(slice_name: str) -> list[str]:
    """Load task IDs for a specific zipper slice (S1-S8 or E1-E2)."""
    split_path = Path("data/processed/zipper_split.json")
    if not split_path.exists():
        print(f"Error: {split_path} not found. Run the zipper splitter first.")
        sys.exit(1)
    with open(split_path) as f:
        splits = json.load(f)

    # Check dev_slices first, then eval_slices
    dev_slices = splits.get("dev_slices", {})
    eval_slices = splits.get("eval_slices", {})
    all_slices = {**dev_slices, **eval_slices}
    available = list(all_slices.keys())
    if slice_name not in all_slices:
        print(f"Error: slice '{slice_name}' not found. Available: {available}")
        sys.exit(1)
    return all_slices[slice_name]


def filter_samples_by_ids(samples: list[Sample], task_ids: list[str]) -> list[Sample]:
    """Filter samples to only include those matching the given task IDs."""
    id_set = set(task_ids)
    return [s for s in samples if s.id in id_set]


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GDPval evaluation with pluggable agent backends"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="claude",
        choices=list(AGENTS.keys()),
        help="Agent backend: claude, gemini, codex (default: claude)",
    )
    parser.add_argument("--model", type=str, default=None, help="Override the agent model")
    parser.add_argument("--n", type=int, default=3, help="Number of tasks to run (default: 3)")
    parser.add_argument("--all", action="store_true", help="Run all 220 tasks")
    parser.add_argument(
        "--slice", type=str, help="Run a specific zipper slice (S1-S8 dev, E1-E2 eval)"
    )
    parser.add_argument(
        "--max-turns", type=int, default=30, help="Max agent turns per task (default: 10)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=32, help="Number of concurrent tasks (default: 32)"
    )
    parser.add_argument(
        "--output", type=str, help="Output directory (default: results/<agent>_<timestamp>/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Use keyword heuristic instead of Gemini judge (faster, less accurate)",
    )
    parser.add_argument(
        "--judge-model", type=str, default=None, help="Override Gemini judge model name"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create agent
    agent_cls = AGENTS[args.agent]
    agent = agent_cls(max_turns=args.max_turns, model=args.model)
    print(f"Agent: {agent.name()}")

    # Load GDPval dataset
    print("Loading GDPval dataset...")
    registry = DatasetRegistry()
    registry.load_dataset("gdpval")
    all_samples = registry.get_samples("gdpval")
    print(f"  Loaded {len(all_samples)} GDPval tasks")

    # Select tasks
    if args.slice:
        task_ids = load_zipper_slice(args.slice)
        samples = filter_samples_by_ids(all_samples, task_ids)
        print(f"  Using zipper slice {args.slice}: {len(samples)} tasks")
    elif args.all:
        samples = all_samples
        print(f"  Running ALL {len(samples)} tasks")
    else:
        sampler = UnifiedSampler(registry, seed=args.seed)
        samples = sampler.sample("gdpval", n=args.n)
        print(f"  Sampled {len(samples)} tasks")

    # Show task overview
    print("\nTasks to evaluate:")
    for i, s in enumerate(samples[:10]):
        occ = s.metadata.get("occupation", "unknown")
        print(f"  {i + 1}. [{occ}] {s.id[:12]}... — {s.prompt[:80]}...")
    if len(samples) > 10:
        print(f"  ... and {len(samples) - 10} more")

    # Set up output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/{args.agent}_{timestamp}")

    # Create workspace inside the output directory so runs don't interfere
    work_dir = output_dir / "workspace"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation — Gemini judge is default, --no-judge for keyword heuristic
    use_judge = not args.no_judge
    evaluator_label = "Gemini judge (frozen)" if use_judge else "keyword heuristic"
    print(
        f"\nStarting evaluation (agent={agent.name()}, max_turns={args.max_turns}, concurrency={args.concurrency})"
    )
    print(f"Evaluator: {evaluator_label}")
    print("=" * 60)

    runner = GDPvalRunner(
        agent=agent,
        working_dir=work_dir,
        use_judge=use_judge,
        judge_model=args.judge_model,
    )

    completed_traces: list = []

    def on_progress(completed: int, total: int, trace: object) -> None:
        completed_traces.append(trace)
        # Compute running stats
        scored = [
            t
            for t in completed_traces
            if hasattr(t, "eval_result") and t.eval_result and t.eval_result.max_score > 0
        ]
        errors = [t for t in completed_traces if hasattr(t, "error") and t.error]
        avg = sum(t.eval_result.normalized_score for t in scored) / len(scored) if scored else 0

        # Get this task's info
        occ = ""
        score_str = ""
        task_id = getattr(trace, "task_id", "")[:12]
        if hasattr(trace, "eval_result") and trace.eval_result and trace.eval_result.max_score > 0:
            score_str = f"{trace.eval_result.normalized_score:.0%}"
        elif hasattr(trace, "error") and trace.error:
            score_str = "ERR"
        for s in samples:
            if s.id[:12] == task_id:
                occ = s.metadata.get("occupation", "")[:30]
                break
        dur = getattr(trace, "duration_s", 0)
        print(
            f"  [{completed:2d}/{total}] {score_str:>5s}  {occ:30s}  {dur:6.0f}s  | avg={avg:.0%} err={len(errors)}",
            flush=True,
        )

    result = await runner.run_batch(
        samples,
        concurrency=args.concurrency,
        progress_callback=on_progress,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Agent:          {agent.name()}")
    print(f"  Tasks run:      {len(result.traces)}")
    print(f"  Completed:      {result.num_completed}")
    print(f"  Errors:         {result.num_errors}")
    print(f"  Avg score:      {result.avg_score:.2%}")
    print(f"  Total time:     {result.total_duration_s:.1f}s")
    print()

    # Per-task breakdown
    print("Per-task scores:")
    for trace in result.traces:
        status = "ERROR" if trace.error else "OK"
        score_str = ""
        if trace.eval_result:
            score_str = f"{trace.eval_result.normalized_score:.0%}"
        occ = ""
        for s in samples:
            if s.id == trace.task_id:
                occ = s.metadata.get("occupation", "")[:25]
                break
        print(
            f"  [{status}] {trace.task_id[:12]}... {occ:25s} {score_str:>5s}  ({trace.duration_s:.1f}s)"
        )

    # Save results
    trace_path, eval_path = runner.save_results(result, output_dir)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  Traces:  {trace_path}")
    print(f"  Eval:    {eval_path}")


if __name__ == "__main__":
    asyncio.run(main())
