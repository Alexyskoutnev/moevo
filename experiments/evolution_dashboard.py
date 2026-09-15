"""Local, read-only dashboard and portable figure export for an evolution run."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from moevo.reporting.aggregate import aggregate_panel
from moevo.reporting.progress import controller_liveness, task_progress
from moevo.reporting.references import reference_comparison
from moevo.reporting.results import export_tables
from moevo.reporting.source import source_view

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "moevo/reporting/dashboard.html"


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def task_outcome(benchmark: str, row: dict | None) -> tuple[str, str]:
    """Show the sample denominator and keep native completion separate from quality."""
    if row is None:
        return "Waiting", ""
    grade = row.get("native_metrics", {}).get("grade", {})
    binary = {
        "finqa",
        "bizfinbench2",
        "amo",
        "putnambench",
        "travelplanner",
        "dsbench",
        "terminal_bench_2",
        "tau3_bench",
    }
    detail = ""
    if benchmark in binary:
        passed = row["score"] == 1
    elif benchmark == "genebench_pro":
        passed = grade.get("passed")
        detail = f"Task quality: {100 * row['score']:.1f}%"
    elif benchmark == "harvey_lab":
        passed = grade.get("all_pass")
        detail = f"{grade.get('n_passed', '?')} of {grade.get('n_criteria', '?')} checks met"
    elif benchmark == "healthbench_professional":
        passed = grade.get("all_criteria_passed")
        detail = f"This task's rubric score: {100 * row['score']:.1f}%"
    elif benchmark == "oragentbench":
        feasible = grade.get("evaluation", {}).get("feasible")
        if isinstance(feasible, bool):
            return f"{int(feasible)} feasible / 1 tested", "One optimization problem"
        passed = None
    else:
        passed = None
    if not isinstance(passed, bool):
        return "1 task tested", detail or f"Task score: {100 * row['score']:.1f}%"
    return f"{int(passed)} passed / 1 tested", detail


def snapshot(
    run_root: Path,
    study_root: Path | None = None,
    *,
    include_source: bool = False,
    observe_process: bool = False,
    reference_roots: list[Path] | None = None,
) -> dict:
    run = read_json(run_root / "run.json", {})
    manifest = read_json(run_root / "search.json", {})
    state = read_json(run_root / "evaluation_state.json", {})
    checkpoints = sorted(run_root.glob("checkpoint_*.json"))
    checkpoint = read_json(checkpoints[-1], {}) if checkpoints else {}
    database = checkpoint.get("database", {})
    objectives = database.get("objectives", run.get("config", {}).get("objectives", []))
    history = []
    seen = {}
    for program in database.get("all_programs", []):
        if program["solution"] in seen:
            continue
        seen[program["solution"]] = program["id"]
        metrics = program["metrics"]
        if any(o not in metrics or not math.isfinite(metrics[o]) for o in objectives):
            continue
        history.append(
            {
                "id": program["id"],
                "candidate_sha256": hashlib.sha256(
                    json.dumps(program["solution"], sort_keys=True).encode()
                ).hexdigest(),
                "step": program["iteration"] + 1 if program.get("parent_id") else 0,
                "metrics": metrics,
                "mean": sum(metrics[o] for o in objectives) / len(objectives),
                "minimum": min(metrics[o] for o in objectives),
            }
        )
    baseline = history[0] if history else None
    current_ids = {
        seen.get(p["solution"], p["id"]) for island in database.get("islands", []) for p in island
    }
    eligible = [p for p in history if p["id"] in current_ids]
    champion = max(
        eligible, key=lambda p: tuple(sorted(p["metrics"][o] for o in objectives)), default=None
    )
    screens = [
        {
            k: e.get(k)
            for k in (
                "screen",
                "delta",
                "improved",
                "audit",
                "confirmed",
                "screen_missed_improvement",
            )
        }
        for e in state.get("events", [])
        if e["stage"] == "screen"
    ]
    # During the initial all-domain evaluation, show measured tasks without inventing a full vector.
    seed_file = run_root / "seed.py"
    seed_hash = (
        hashlib.sha256(json.dumps(seed_file.read_text(), sort_keys=True).encode()).hexdigest()
        if seed_file.exists()
        else None
    )
    seed_rows = {
        r["task_id"]: r
        for r in state.get("cache", {}).values()
        if r.get("candidate_sha256") == seed_hash
    }
    benchmarks = []
    for task in manifest.get("tasks", []):
        row = seed_rows.get(task["id"])
        outcome, detail = task_outcome(task["benchmark"], row)
        benchmarks.append(
            {
                "benchmark": task["benchmark"],
                "domain": task["objective"],
                "seed_score": row.get("score") if row else None,
                "seed_outcome": outcome,
                "seed_detail": detail,
            }
        )
    panel_ids = set(manifest.get("confirmation_ids", []))
    panel = [t for t in manifest.get("tasks", []) if t["id"] in panel_ids]
    aggregate = aggregate_panel(
        panel,
        state.get("cache", {}),
        seed_hash,
        champion["candidate_sha256"] if champion else None,
    )
    evolved = [point for point in history if point["step"] > 0]
    comparison = (
        "pending"
        if not evolved
        else "starting_agent_retained"
        if champion and baseline and champion["id"] == baseline["id"]
        else "evolved_agent_selected"
    )
    # A screen is charged before it finishes; show that work as active, not completed.
    screens_started = state.get("screens", len(screens))
    active = None
    if run.get("status") == "running" and screens_started:
        latest = screens[-1] if screens else None
        if latest is None or screens_started > (latest.get("screen") or 0):
            active = {"screen": screens_started, "stage": "quick_check"}
        elif not latest["confirmed"] and (latest["improved"] or latest["audit"]):
            active = {"screen": screens_started, "stage": "full_test"}
    return {
        "title": "SuperHarness · one epoch",
        "status": run.get("status", "waiting"),
        "purpose": run.get("purpose", "Search scores; not held-out benchmark results"),
        "solver": run.get("policy", {}).get("model", "gpt-6-astra"),
        "judge": run.get("policy", {}).get("judge_model", "gpt-5.6-terra"),
        "objectives": objectives,
        "history": history,
        "baseline": baseline,
        "champion": champion,
        "screens": screens,
        "screens_started": screens_started,
        "active_evaluation": active,
        "complete_evolved_versions": len(evolved),
        "comparison_status": comparison,
        "benchmarks": benchmarks,
        "aggregate": aggregate,
        "split": manifest.get("split", "unknown"),
        "protocol": manifest.get("protocol"),
        "task_evaluations": state.get("task_evaluations", 0),
        "task_budget": run.get("config", {}).get("max_task_evaluations"),
        "planned_steps": run.get("config", {}).get("iterations"),
        "search_steps_completed": max(0, checkpoint.get("iteration", -1) + 1),
        "candidate_attempt_cap": run.get("config", {}).get("iterations", 0)
        * (run.get("config", {}).get("retry_attempts", 0) + 1),
        "task_errors": sum(e["stage"] == "task_error" for e in state.get("events", [])),
        "excluded": run.get("excluded", {}),
        "error": run.get("error"),
        "next_study": read_json(study_root / "summary.json", None) if study_root else None,
        "source_view": source_view(run_root, database, state) if include_source else None,
        "progress": task_progress(
            run_root,
            run,
            manifest,
            state,
            database,
            process=controller_liveness(run_root) if observe_process else None,
        ),
        "run_label": (
            "Fresh-task first-slice test"
            if run.get("execution_role") == "search"
            else "Earlier fixture diagnostic"
            if "existing fixtures" in run.get("purpose", "")
            else run_root.name
        ),
        "reference_baselines": reference_comparison(
            manifest, run.get("policy", {}), reference_roots or [], observe_process=observe_process
        ),
    }


def charts(data: dict, output: Path | None = None) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.projections.polar import PolarAxes

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#d1d9dc",
            "axes.labelcolor": "#53646b",
            "text.color": "#203139",
            "xtick.color": "#53646b",
            "ytick.color": "#53646b",
        }
    )
    figures = {}

    def save(name, figure):
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor="white")
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{name}.png").write_bytes(buffer.getvalue())
            for extension in ("pdf", "svg"):
                figure.savefig(
                    output / f"{name}.{extension}", bbox_inches="tight", facecolor="white"
                )
        plt.close(figure)
        figures[name] = base64.b64encode(buffer.getvalue()).decode()

    history, objectives = data["history"], data["objectives"]
    if not history:
        return figures
    x = [p["step"] for p in history]
    figure, ax = plt.subplots(figsize=(11, 3.5), layout="constrained")
    ax.plot(x, [100 * p["mean"] for p in history], "o-", color="#197c79", label="Average score")
    ax.plot(
        x,
        [100 * p["minimum"] for p in history],
        "s-",
        color="#cb7656",
        label="Lowest task-area score",
    )
    incumbent = []
    best = history[0]
    for point in history:
        if tuple(sorted(point["metrics"][o] for o in objectives)) > tuple(
            sorted(best["metrics"][o] for o in objectives)
        ):
            best = point
        incumbent.append(100 * best["minimum"])
    ax.step(
        x,
        incumbent,
        where="post",
        linestyle="--",
        color="#394554",
        label="Best lowest-area score so far",
    )
    ax.set(
        xlabel="Agent version · 0 is the starting agent",
        ylabel="Mean normalized score (%)",
        ylim=(-3, 103),
    )
    ax.set_xticks(range((data["planned_steps"] or max(x)) + 1))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower left", ncol=1, frameon=False, fontsize=8)
    save("progress", figure)

    ncols = 3
    figure, axes = plt.subplots(
        math.ceil(len(objectives) / ncols),
        ncols,
        figsize=(12, 2.3 * math.ceil(len(objectives) / ncols)),
        squeeze=False,
        layout="constrained",
    )
    baseline = data["baseline"]
    for i, objective in enumerate(objectives):
        ax = axes.flat[i]
        ax.axhline(
            100 * baseline["metrics"][objective], color="#9aaab2", linestyle=":", linewidth=1
        )
        ax.plot(
            x, [100 * p["metrics"][objective] for p in history], "o-", color="#197c79", markersize=4
        )
        ax.set(
            title=objective.replace("_", " ").capitalize(),
            ylim=(-3, 103),
            xlabel="Agent version",
            ylabel="Score (%)",
        )
        ax.set_xticks(range((data["planned_steps"] or max(x)) + 1))
        ax.grid(axis="y", alpha=0.18)
    for ax in list(axes.flat)[len(objectives) :]:
        ax.set_visible(False)
    save("domains", figure)

    figure, ax = plt.subplots(
        figsize=(7, 6), subplot_kw={"projection": "polar"}, layout="constrained"
    )
    angles = np.linspace(0, 2 * np.pi, len(objectives), endpoint=False).tolist()
    angles += angles[:1]
    plotted = set()
    for label, point, color, style in [
        ("Starting agent", baseline, "#9aaab2", "--"),
        ("Latest version", history[-1], "#cb7656", "-"),
        ("Best all-round version", data["champion"], "#197c79", "-"),
    ]:
        if point is None:
            continue
        if point["id"] in plotted:
            continue
        plotted.add(point["id"])
        values = [100 * point["metrics"][o] for o in objectives]
        ax.plot(angles, values + values[:1], style, label=label, color=color, linewidth=1.8)
    assert isinstance(ax, PolarAxes)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1], [o.replace("_", "\n") for o in objectives], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100], ["25", "50", "75", "100"], fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=8)
    save("radar", figure)
    return figures


def document(data: dict, live: bool) -> str:
    payload = json.dumps({**data, "figures": charts(data)}, allow_nan=False).replace("<", "\\u003c")
    return (
        TEMPLATE.read_text()
        .replace("__INITIAL_DATA__", payload)
        .replace("__LIVE__", str(live).lower())
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--study", type=Path, help="Show the separately prepared eight-slice study")
    parser.add_argument("--related-url", help="Link to the other local run dashboard")
    parser.add_argument("--related-label", default="Other run")
    parser.add_argument(
        "--reference-run",
        type=Path,
        action="append",
        default=[],
        help="Show a task-only Codex or matched seed reference run",
    )
    parser.add_argument("--export", type=Path)
    parser.add_argument(
        "--export-results", type=Path, help="Export percentage tables and PDF/SVG/PNG figures"
    )
    args = parser.parse_args()
    run_root = args.run.resolve()
    if args.export_results:
        data = snapshot(run_root, args.study, include_source=True)
        export_tables(data, args.export_results)
        charts(data, args.export_results)
        (args.export_results / "dashboard.html").write_text(document(data, False))
        print(args.export_results.resolve())
        return
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(document(snapshot(run_root, args.study, include_source=True), False))
        print(args.export.resolve())
        return
    lock = threading.Lock()
    cache: dict[str, object] = {"signature": None, "figures": {}}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in {"/", "/data"}:
                self.send_error(404)
                return
            with lock:
                data = snapshot(
                    run_root,
                    args.study,
                    include_source=True,
                    observe_process=True,
                    reference_roots=args.reference_run,
                )
                if args.related_url:
                    data["related_run"] = {"url": args.related_url, "label": args.related_label}
                # Activity changes independently of scientific figures; avoid redrawing
                # every chart whenever a worker writes a log or finishes a partial task.
                signature = json.dumps(
                    {
                        key: data[key]
                        for key in (
                            "history",
                            "objectives",
                            "baseline",
                            "champion",
                            "planned_steps",
                        )
                    },
                    sort_keys=True,
                )
                if signature != cache["signature"]:
                    cache.update(signature=signature, figures=charts(data))
                data["figures"] = cache["figures"]
                if self.path == "/data":
                    body = json.dumps(data, allow_nan=False).encode()
                    mime = "application/json"
                else:
                    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
                    body = (
                        TEMPLATE.read_text()
                        .replace("__INITIAL_DATA__", payload)
                        .replace("__LIVE__", "true")
                        .encode()
                    )
                    mime = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    print(f"Dashboard: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
