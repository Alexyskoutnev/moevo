"""Visualization functions for GDPval evaluation results.

Produces the figures used in the MOEvo paper:
- Self-improvement trajectory across iterations
- Per-slice baseline scores
- Per-sector breakdown
- Failure taxonomy
- Model comparison charts

Usage:
    from moevo.eval.visualize import ResultsViz

    viz = ResultsViz("results/")
    viz.trajectory()           # Fig 7: score across iterations
    viz.slice_comparison()     # Table 3: per-slice bar chart
    viz.sector_breakdown()     # Fig 8: per-sector grouped bars
    viz.failure_taxonomy()     # Fig 10: points lost by category
    viz.model_comparison()     # Compare multiple models
    viz.dashboard()            # All-in-one summary
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False

logger = logging.getLogger(__name__)

# Paper-style defaults (only if matplotlib is available)
if _HAS_PLOTTING:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

# Color palette
COLORS = {
    "primary": "#2563EB",  # Blue
    "secondary": "#7C3AED",  # Purple
    "success": "#059669",  # Green
    "warning": "#D97706",  # Amber
    "danger": "#DC2626",  # Red
    "neutral": "#6B7280",  # Gray
    "light": "#E5E7EB",  # Light gray
    "baseline": "#93C5FD",  # Light blue
    "improved": "#34D399",  # Light green
}

SECTOR_COLORS = [
    "#2563EB",
    "#7C3AED",
    "#059669",
    "#D97706",
    "#DC2626",
    "#0891B2",
    "#4F46E5",
    "#CA8A04",
    "#BE185D",
]


@dataclass
class RunResult:
    """Parsed evaluation result from a single run."""

    run_dir: str
    timestamp: str
    label: str = ""
    avg_score: float = 0.0
    num_tasks: int = 0
    num_completed: int = 0
    num_errors: int = 0
    total_duration_s: float = 0.0
    scores: list[dict[str, Any]] = field(default_factory=list)

    @property
    def task_scores(self) -> list[float]:
        return [s.get("normalized_score", 0.0) for s in self.scores if s.get("error") is None]

    @property
    def task_ids(self) -> list[str]:
        return [s["task_id"] for s in self.scores]

    def scores_by_occupation(self, samples: list[Any] | None = None) -> dict[str, list[float]]:
        """Group scores by occupation (requires samples for metadata lookup)."""
        if not samples:
            return {}
        id_to_occ = {s.id: s.metadata.get("occupation", "Unknown") for s in samples}
        by_occ: dict[str, list[float]] = {}
        for s in self.scores:
            occ = id_to_occ.get(s["task_id"], "Unknown")
            by_occ.setdefault(occ, []).append(s.get("normalized_score", 0.0))
        return by_occ

    def scores_by_sector(self, samples: list[Any] | None = None) -> dict[str, list[float]]:
        """Group scores by GDP sector (requires samples for metadata lookup)."""
        if not samples:
            return {}
        id_to_sector = {s.id: s.metadata.get("sector", "Unknown") for s in samples}
        by_sector: dict[str, list[float]] = {}
        for s in self.scores:
            sector = id_to_sector.get(s["task_id"], "Unknown")
            by_sector.setdefault(sector, []).append(s.get("normalized_score", 0.0))
        return by_sector


def plot_baseline_trajectory(
    trajectory_path: str | Path,
    title: str = "Baseline Score Trajectory",
    save_path: str | None = None,
) -> Any:
    """Plot per-agent scores across slices from a trajectory.json file.

    Produces a line chart with one line per agent (claude, gemini, codex)
    showing their score on each slice (S1-S8, E1, E2). A vertical dashed
    line separates dev slices from eval slices.

    Args:
        trajectory_path: Path to trajectory.json from a baseline run.
        title: Plot title.
        save_path: If set, save figure to this path instead of showing.

    Returns:
        matplotlib Figure, or None if plotting deps are missing.
    """
    if not _HAS_PLOTTING:
        logger.warning("matplotlib/numpy not installed - plotting disabled")
        return None

    path = Path(trajectory_path)
    if path.is_dir():
        path = path / "trajectory.json"
    with open(path) as f:
        trajectory = json.load(f)

    agents = trajectory.get("agents", {})
    if not agents:
        logger.warning("No agent data in %s", path)
        return None

    slice_order = [f"S{i}" for i in range(1, 9)] + ["E1", "E2"]
    agent_colors = {
        "claude": COLORS["primary"],
        "gemini": COLORS["warning"],
        "codex": COLORS["secondary"],
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for agent_name, agent_data in agents.items():
        scores_dict = agent_data.get("scores", {})
        x_vals = []
        y_vals = []
        for i, s in enumerate(slice_order):
            if s in scores_dict:
                x_vals.append(i)
                y_vals.append(scores_dict[s]["avg_score"] * 100)

        if not x_vals:
            continue

        color = agent_colors.get(agent_name, COLORS["neutral"])
        ax.plot(
            x_vals,
            y_vals,
            "o-",
            color=color,
            linewidth=2.5,
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=2.5,
            label=agent_name,
            zorder=3,
        )

        # Score labels
        for xi, yi in zip(x_vals, y_vals, strict=False):
            ax.annotate(
                f"{yi:.1f}%",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

    # Vertical line between dev and eval slices
    ax.axvline(x=7.5, color=COLORS["neutral"], linestyle="--", alpha=0.5)
    ax.text(
        6,
        ax.get_ylim()[1] - 2,
        "dev",
        ha="center",
        fontsize=10,
        color=COLORS["neutral"],
        fontstyle="italic",
    )
    ax.text(
        8.5,
        ax.get_ylim()[1] - 2,
        "eval",
        ha="center",
        fontsize=10,
        color=COLORS["neutral"],
        fontstyle="italic",
    )

    ax.set_xticks(range(len(slice_order)))
    ax.set_xticklabels(slice_order, fontsize=11)
    ax.set_ylabel("GDPval Average Score (%)")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info("Saved trajectory plot to %s", save_path)
    else:
        plt.show()
    return fig


class ResultsViz:
    """Visualization toolkit for GDPval evaluation results.

    Scans a results directory for eval.json files and provides
    paper-quality plotting functions.
    """

    def __init__(self, results_dir: str | Path = "results") -> None:
        self._results_dir = Path(results_dir)
        self._runs: list[RunResult] = []
        self._load_all_runs()

    @staticmethod
    def _require_plotting() -> bool:
        """Check if plotting deps are available. Warn once if not."""
        if not _HAS_PLOTTING:
            logger.warning(
                "matplotlib/numpy not installed - plotting disabled. "
                "Install with: pip install matplotlib numpy"
            )
            return False
        return True

    def _load_all_runs(self) -> None:
        """Discover and load all eval.json files in the results directory."""
        self._runs = []
        for eval_file in sorted(self._results_dir.glob("*/eval.json")):
            try:
                data = json.loads(eval_file.read_text())
                run_dir = eval_file.parent.name
                # Extract timestamp from dir name (gdpval_YYYYMMDD_HHMMSS)
                parts = run_dir.split("_")
                timestamp = "_".join(parts[1:]) if len(parts) >= 3 else run_dir

                run = RunResult(
                    run_dir=run_dir,
                    timestamp=timestamp,
                    avg_score=data["summary"]["avg_score"],
                    num_tasks=data["summary"]["num_tasks"],
                    num_completed=data["summary"].get("num_completed", 0),
                    num_errors=data["summary"].get("num_errors", 0),
                    total_duration_s=data["summary"].get("total_duration_s", 0),
                    scores=data.get("scores", []),
                )
                self._runs.append(run)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Skipping %s: %s", eval_file, e)

        logger.info("Loaded %d runs from %s", len(self._runs), self._results_dir)

    def reload(self) -> None:
        """Reload results from disk."""
        self._load_all_runs()

    @property
    def runs(self) -> list[RunResult]:
        return list(self._runs)

    def label_runs(self, labels: dict[str, str]) -> None:
        """Assign labels to runs by directory name or index.

        Args:
            labels: Dict mapping run_dir (or index as str) to label.
                    e.g. {"gdpval_20260310_182156": "Baseline v1"}
        """
        for run in self._runs:
            if run.run_dir in labels:
                run.label = labels[run.run_dir]
        for idx_str, label in labels.items():
            if idx_str.isdigit():
                idx = int(idx_str)
                if 0 <= idx < len(self._runs):
                    self._runs[idx].label = label

    def list_runs(self) -> None:
        """Print a summary table of all loaded runs."""
        print(
            f"{'#':>3}  {'Directory':<35} {'Label':<20} {'Score':>7} {'Tasks':>5} {'Errors':>6} {'Time':>8}"
        )
        print("─" * 90)
        for i, run in enumerate(self._runs):
            label = run.label or "-"
            print(
                f"{i:>3}  {run.run_dir:<35} {label:<20} "
                f"{run.avg_score:>6.1%} {run.num_tasks:>5} "
                f"{run.num_errors:>6} {run.total_duration_s:>7.0f}s"
            )

    # ─── Plotting Functions ───────────────────────────────────────────

    def trajectory(
        self,
        runs: list[RunResult] | None = None,
        title: str = "Self-Improvement Trajectory",
        save_path: str | None = None,
        show_delta: bool = True,
    ) -> Any:
        """Plot score trajectory across iterations (Fig 7 style).

        Args:
            runs: Specific runs to plot (default: all runs in order).
            title: Plot title.
            save_path: If set, save figure to this path.
            show_delta: Show delta labels between iterations.
        """
        if not self._require_plotting():
            return None
        runs = runs or self._runs
        if not runs:
            print("No runs loaded.")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        x = list(range(len(runs)))
        scores = [r.avg_score * 100 for r in runs]
        labels = [r.label or f"Run {i}" for i, r in enumerate(runs)]

        # Main line + markers
        ax.plot(
            x,
            scores,
            "o-",
            color=COLORS["primary"],
            linewidth=2.5,
            markersize=10,
            markerfacecolor="white",
            markeredgewidth=2.5,
            zorder=3,
        )

        # Score labels
        for _i, (xi, score) in enumerate(zip(x, scores, strict=False)):
            ax.annotate(
                f"{score:.1f}%",
                (xi, score),
                textcoords="offset points",
                xytext=(0, 15),
                ha="center",
                fontweight="bold",
                fontsize=11,
                color=COLORS["primary"],
            )

        # Delta labels
        if show_delta and len(scores) > 1:
            for i in range(1, len(scores)):
                delta = scores[i] - scores[i - 1]
                color = COLORS["success"] if delta > 0 else COLORS["danger"]
                sign = "+" if delta > 0 else ""
                ax.annotate(
                    f"{sign}{delta:.1f}%",
                    ((x[i] + x[i - 1]) / 2, (scores[i] + scores[i - 1]) / 2),
                    textcoords="offset points",
                    xytext=(0, -20),
                    ha="center",
                    fontsize=9,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.1),
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("GDPval Average Score (%)")
        ax.set_title(title, fontweight="bold", pad=15)
        ax.set_ylim(max(0, min(scores) - 10), min(100, max(scores) + 10))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
            print(f"Saved to {save_path}")
        plt.show()
        return fig

    def slice_comparison(
        self,
        run: RunResult | None = None,
        samples: list[Any] | None = None,
        title: str = "Frozen Judge Baseline Scores by Dev Slice",
        save_path: str | None = None,
    ) -> Any:
        """Per-slice bar chart (Table 3 / Fig 7 style).

        If samples are provided, groups tasks by zipper slice.
        Otherwise, shows per-task scores.
        """
        if not self._require_plotting():
            return None
        run = run or (self._runs[-1] if self._runs else None)
        if not run:
            print("No runs loaded.")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        task_scores = []
        task_labels = []
        for s in run.scores:
            if s.get("error") is not None:
                continue
            score = s.get("normalized_score", 0.0) * 100
            task_id = s["task_id"][:12]
            task_scores.append(score)
            task_labels.append(task_id)

        if not task_scores:
            print("No scored tasks in this run.")
            return fig

        x = np.arange(len(task_scores))
        colors = [
            COLORS["success"] if s >= 90 else COLORS["warning"] if s >= 70 else COLORS["danger"]
            for s in task_scores
        ]

        bars = ax.bar(x, task_scores, color=colors, edgecolor="white", linewidth=0.5)

        # Score labels on bars
        for bar, score in zip(bars, task_scores, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{score:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Mean line
        mean_score = np.mean(task_scores)
        ax.axhline(mean_score, color=COLORS["primary"], linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(
            len(task_scores) - 0.5,
            mean_score + 1.5,
            f"Mean: {mean_score:.1f}%",
            ha="right",
            color=COLORS["primary"],
            fontweight="bold",
            fontsize=10,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Score (%)")
        ax.set_title(title, fontweight="bold", pad=15)
        ax.set_ylim(0, 110)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
            print(f"Saved to {save_path}")
        plt.show()
        return fig

    def sector_breakdown(
        self,
        runs: list[RunResult] | None = None,
        samples: list[Any] | None = None,
        title: str = "Self-Improvement by GDP Sector",
        save_path: str | None = None,
    ) -> Any:
        """Per-sector grouped bar chart (Fig 8 style).

        Requires samples to map task_id → sector.
        Compares first run (baseline) vs last run (final).
        """
        if not self._require_plotting():
            return None
        if not samples:
            print("Provide samples (from DatasetRegistry) to map tasks to sectors.")
            return None

        runs = runs or self._runs
        if not runs:
            print("No runs loaded.")
            return None

        baseline = runs[0]
        final = runs[-1] if len(runs) > 1 else runs[0]

        baseline_sectors = baseline.scores_by_sector(samples)
        final_sectors = final.scores_by_sector(samples)

        all_sectors = sorted(set(list(baseline_sectors.keys()) + list(final_sectors.keys())))

        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(all_sectors))
        width = 0.35

        baseline_means = [np.mean(baseline_sectors.get(s, [0])) * 100 for s in all_sectors]
        final_means = [np.mean(final_sectors.get(s, [0])) * 100 for s in all_sectors]
        counts = [len(final_sectors.get(s, baseline_sectors.get(s, []))) for s in all_sectors]

        ax.bar(
            x - width / 2,
            baseline_means,
            width,
            label=baseline.label or "Baseline",
            color=COLORS["baseline"],
            edgecolor="white",
        )
        ax.bar(
            x + width / 2,
            final_means,
            width,
            label=final.label or "Final",
            color=COLORS["improved"],
            edgecolor="white",
        )

        # Delta labels
        for i, (bm, fm) in enumerate(zip(baseline_means, final_means, strict=False)):
            delta = fm - bm
            if delta != 0:
                sign = "+" if delta > 0 else ""
                ax.text(
                    i,
                    max(bm, fm) + 2,
                    f"{sign}{delta:.1f}%",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color=COLORS["success"] if delta > 0 else COLORS["danger"],
                )

        # Task counts
        for i, n in enumerate(counts):
            ax.text(i, -4, f"n={n}", ha="center", fontsize=8, color=COLORS["neutral"])

        # Shorten long sector names
        short_names = [
            s.replace(" and ", " & ").replace(", ", ",\n") if len(s) > 20 else s
            for s in all_sectors
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Average Score (%)")
        ax.set_title(title, fontweight="bold", pad=15)
        ax.set_ylim(0, 115)
        ax.legend(loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
            print(f"Saved to {save_path}")
        plt.show()
        return fig

    def failure_taxonomy(
        self,
        run: RunResult | None = None,
        title: str = "Points Lost by Failure Category",
        save_path: str | None = None,
    ) -> Any:
        """Horizontal bar chart of points lost per rubric criterion (Fig 10 style).

        Shows which criteria are most commonly failed.
        """
        if not self._require_plotting():
            return None
        run = run or (self._runs[-1] if self._runs else None)
        if not run:
            print("No runs loaded.")
            return None

        # Aggregate failed criteria across tasks
        failures: dict[str, float] = {}
        for s in run.scores:
            breakdown = s.get("rubric_breakdown", {})
            for criterion, points in breakdown.items():
                if points == 0:
                    failures[criterion] = failures.get(criterion, 0) + 1

        if not failures:
            print("No failures found in this run.")
            return None

        # Sort by frequency, take top 20
        sorted_failures = sorted(failures.items(), key=lambda x: x[1], reverse=True)[:20]
        criteria = [f[:60] + "..." if len(f) > 60 else f for f, _ in sorted_failures]
        counts = [c for _, c in sorted_failures]

        fig, ax = plt.subplots(figsize=(12, max(6, len(criteria) * 0.4)))

        y = np.arange(len(criteria))
        ax.barh(y, counts, color=COLORS["danger"], alpha=0.8, edgecolor="white")

        for i, count in enumerate(counts):
            ax.text(count + 0.2, i, f"{count:.0f}", va="center", fontsize=9)

        ax.set_yticks(y)
        ax.set_yticklabels(criteria, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Number of Tasks Failed")
        ax.set_title(title, fontweight="bold", pad=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
            print(f"Saved to {save_path}")
        plt.show()
        return fig

    def model_comparison(
        self,
        runs: list[RunResult] | None = None,
        title: str = "Model Comparison on GDPval",
        save_path: str | None = None,
    ) -> Any:
        """Grouped bar chart comparing multiple models/runs.

        Each run is a separate bar. Use label_runs() to set display names.
        """
        if not self._require_plotting():
            return None
        runs = runs or self._runs
        if not runs:
            print("No runs loaded.")
            return None

        fig, ax = plt.subplots(figsize=(max(8, len(runs) * 2), 6))

        labels = [r.label or r.run_dir for r in runs]
        scores = [r.avg_score * 100 for r in runs]
        task_counts = [r.num_tasks for r in runs]

        x = np.arange(len(runs))
        colors = [
            COLORS["primary"]
            if i == 0
            else COLORS["improved"]
            if s == max(scores)
            else COLORS["neutral"]
            for i, s in enumerate(scores)
        ]

        bars = ax.bar(x, scores, color=colors, edgecolor="white", width=0.6)

        for bar, score, n in zip(bars, scores, task_counts, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{score:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                2,
                f"n={n}",
                ha="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("GDPval Average Score (%)")
        ax.set_title(title, fontweight="bold", pad=15)
        ax.set_ylim(0, min(110, max(scores) + 15))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
            print(f"Saved to {save_path}")
        plt.show()
        return fig

    def criteria_heatmap(
        self,
        run: RunResult | None = None,
        title: str = "Rubric Criteria Scores",
        save_path: str | None = None,
    ) -> Any:
        """Heatmap of per-task per-criterion scores.

        Rows = tasks, columns = criteria. Green = met, red = failed.
        """
        if not self._require_plotting():
            return None
        run = run or (self._runs[-1] if self._runs else None)
        if not run:
            print("No runs loaded.")
            return None

        # Collect all criteria across tasks
        all_criteria: set[str] = set()
        for s in run.scores:
            all_criteria.update(s.get("rubric_breakdown", {}).keys())

        if not all_criteria:
            print("No rubric breakdown data available.")
            return None

        criteria_list = sorted(all_criteria)
        task_ids = [s["task_id"][:12] for s in run.scores if s.get("rubric_breakdown")]

        # Build matrix
        matrix = []
        for s in run.scores:
            breakdown = s.get("rubric_breakdown", {})
            if not breakdown:
                continue
            row = [1.0 if breakdown.get(c, 0) > 0 else 0.0 for c in criteria_list]
            matrix.append(row)

        if not matrix:
            print("No rubric data to plot.")
            return None

        matrix_np = np.array(matrix)

        fig, ax = plt.subplots(
            figsize=(max(10, len(criteria_list) * 0.3), max(4, len(task_ids) * 0.5))
        )

        from matplotlib.colors import ListedColormap

        cmap = ListedColormap([COLORS["danger"], COLORS["success"]])

        ax.imshow(matrix_np, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        ax.set_yticks(range(len(task_ids)))
        ax.set_yticklabels(task_ids, fontsize=8)
        ax.set_xticks(range(len(criteria_list)))
        ax.set_xticklabels([c[:30] for c in criteria_list], rotation=90, fontsize=6)
        ax.set_title(title, fontweight="bold", pad=15)

        # Legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=COLORS["success"], label="Met"),
            Patch(facecolor=COLORS["danger"], label="Failed"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
            print(f"Saved to {save_path}")
        plt.show()
        return fig

    def dashboard(
        self,
        save_path: str | None = None,
    ) -> Any:
        """All-in-one summary dashboard with 4 subplots."""
        if not self._require_plotting():
            return None
        if not self._runs:
            print("No runs loaded.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("GDPval Evaluation Dashboard", fontsize=16, fontweight="bold", y=0.98)

        # ─── Top Left: Score Trajectory ───
        ax = axes[0, 0]
        scores = [r.avg_score * 100 for r in self._runs]
        labels = [r.label or f"Run {i}" for i, r in enumerate(self._runs)]
        ax.plot(
            range(len(scores)),
            scores,
            "o-",
            color=COLORS["primary"],
            linewidth=2,
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=2,
        )
        for i, s in enumerate(scores):
            ax.annotate(
                f"{s:.1f}%",
                (i, s),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontweight="bold",
                fontsize=9,
            )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Score (%)")
        ax.set_title("Score Trajectory", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ─── Top Right: Latest Run Per-Task Scores ───
        ax = axes[0, 1]
        latest = self._runs[-1]
        task_scores = [
            s.get("normalized_score", 0) * 100 for s in latest.scores if s.get("error") is None
        ]
        if task_scores:
            colors = [
                COLORS["success"] if s >= 90 else COLORS["warning"] if s >= 70 else COLORS["danger"]
                for s in task_scores
            ]
            ax.bar(range(len(task_scores)), task_scores, color=colors, edgecolor="white")
            ax.axhline(np.mean(task_scores), color=COLORS["primary"], linestyle="--", alpha=0.7)
            ax.set_ylabel("Score (%)")
            ax.set_title(f"Latest Run: Per-Task Scores (n={len(task_scores)})", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No scored tasks", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Latest Run: Per-Task Scores", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        # ─── Bottom Left: Duration per Run ───
        ax = axes[1, 0]
        durations = [r.total_duration_s / 60 for r in self._runs]
        ax.bar(range(len(durations)), durations, color=COLORS["neutral"], edgecolor="white")
        for i, d in enumerate(durations):
            ax.text(i, d + 0.2, f"{d:.1f}m", ha="center", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Duration (minutes)")
        ax.set_title("Evaluation Duration", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        # ─── Bottom Right: Summary Stats ───
        ax = axes[1, 1]
        ax.axis("off")
        latest = self._runs[-1]
        stats_text = (
            f"Total Runs:     {len(self._runs)}\n"
            f"Latest Score:   {latest.avg_score:.1%}\n"
            f"Best Score:     {max(r.avg_score for r in self._runs):.1%}\n"
            f"Total Tasks:    {sum(r.num_tasks for r in self._runs)}\n"
            f"Total Errors:   {sum(r.num_errors for r in self._runs)}\n"
            f"Total Time:     {sum(r.total_duration_s for r in self._runs) / 60:.1f} min\n"
        )
        if len(self._runs) > 1:
            delta = (self._runs[-1].avg_score - self._runs[0].avg_score) * 100
            sign = "+" if delta > 0 else ""
            stats_text += f"\nImprovement:    {sign}{delta:.1f} pp"

        ax.text(
            0.1,
            0.9,
            "Summary Statistics",
            fontsize=14,
            fontweight="bold",
            transform=ax.transAxes,
            va="top",
        )
        ax.text(
            0.1,
            0.75,
            stats_text,
            fontsize=12,
            fontfamily="monospace",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["light"], alpha=0.5),
        )

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            fig.savefig(save_path)
            print(f"Saved to {save_path}")
        plt.show()
        return fig
