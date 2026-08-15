"""Generate ICML-quality Pareto front evolution scatter plot.

Creates a figure similar to AgentBreeder Figure 2, showing how the
capability-safety Pareto front evolves across iterations and slices.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Colors
SEED_COLOR = "#B0BEC5"  # grey for seed
EARLY_COLOR = "#90CAF9"  # light blue for early iterations
LATE_COLOR = "#1565C0"  # dark blue for late iterations
FRONT_COLOR = "#00897B"  # teal for final Pareto front
BASELINE_CODEX = "#546E7A"  # dark grey
BASELINE_CLAUDE = "#78909C"  # medium grey
CARRIED_COLOR = "#FF6F00"  # orange for carried-forward point
FRONT_FILL = "#E0F2F1"  # light teal fill


def load_all_programs(base_dir):
    """Load all programs from checkpoint files across slices."""
    programs = []
    for s in range(1, 9):
        cp_dir = Path(base_dir) / f"moevo_S{s}"
        if not cp_dir.is_dir():
            continue
        cps = sorted([f for f in os.listdir(cp_dir) if f.startswith("checkpoint_")])
        if not cps:
            continue
        with open(cp_dir / cps[-1]) as f:
            d = json.load(f)
        db = d["database"]
        seen = set()
        for source in [db.get("all_programs", [])] + db.get("islands", []):
            for p in source:
                pid = p["id"][:8]
                if pid in seen:
                    continue
                seen.add(pid)
                m = p.get("metrics", {})
                programs.append(
                    {
                        "slice": f"S{s}",
                        "slice_num": s,
                        "id": pid,
                        "gdpval": m.get("gdpval_score", 0),
                        "safety": m.get("safety_score", 0),
                        "iteration": p.get("iteration", -1),
                    }
                )
    return programs


def load_trajectory(base_dir):
    """Load trajectory to get carry-forward points."""
    traj_path = Path(base_dir) / "trajectory.json"
    if not traj_path.exists():
        return []
    with open(traj_path) as f:
        return json.load(f)


def compute_pareto_front(points):
    """Compute 2D Pareto front (maximize both objectives)."""
    if not points:
        return []
    pts = np.array(points)
    front = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i != j and q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    # Sort by first objective
    front.sort(key=lambda i: pts[i][0])
    return front


def plot_single_slice(ax, programs, trajectory_entry, slice_name, baselines):
    """Plot one slice's evolution on an axis."""
    progs = [p for p in programs if p["slice"] == slice_name]
    if not progs:
        return

    max_iter = max(p["iteration"] for p in progs)

    # Color by iteration (lighter = earlier, darker = later)
    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=-1, vmax=max_iter + 1)

    # Plot seed (iteration 0)
    seeds = [p for p in progs if p["iteration"] == 0]
    others = [p for p in progs if p["iteration"] > 0]

    # Plot evolved points with iteration coloring
    for p in others:
        color = cmap(norm(p["iteration"]))
        ax.scatter(
            p["gdpval"] * 100,
            p["safety"] * 100,
            c=[color],
            s=50,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            zorder=3,
        )

    # Plot seeds
    for p in seeds:
        ax.scatter(
            p["gdpval"] * 100,
            p["safety"] * 100,
            c=[SEED_COLOR],
            s=60,
            marker="D",
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
            zorder=2,
            label="Seed" if p == seeds[0] else "",
        )

    # Compute and shade Pareto front
    all_points = [(p["gdpval"] * 100, p["safety"] * 100) for p in progs]
    front_idx = compute_pareto_front(all_points)
    if front_idx:
        front_pts = np.array([all_points[i] for i in front_idx])
        # Sort for step plot
        order = np.argsort(front_pts[:, 0])
        front_pts = front_pts[order]

        # Shade dominated region
        # Create step function for shading
        x_shade = [0]
        y_shade = [0]
        for pt in front_pts:
            x_shade.extend([pt[0], pt[0]])
            y_shade.extend([y_shade[-1], pt[1]])
        x_shade.append(0)
        y_shade.append(0)
        ax.fill(x_shade, y_shade, color=FRONT_FILL, alpha=0.4, zorder=1)

        # Plot front line
        ax.plot(
            front_pts[:, 0],
            front_pts[:, 1],
            "-",
            color=FRONT_COLOR,
            linewidth=2,
            alpha=0.8,
            zorder=4,
        )
        ax.scatter(
            front_pts[:, 0],
            front_pts[:, 1],
            c=FRONT_COLOR,
            s=90,
            marker="*",
            edgecolors="white",
            linewidth=0.5,
            zorder=5,
        )

    # Mark carried-forward point
    if trajectory_entry and "pareto" in trajectory_entry:
        front_data = trajectory_entry["pareto"].get("front", [])
        if front_data:
            # Find geometric-mean best
            best = max(front_data, key=lambda p: p["gdpval"] * p["safety"])
            ax.scatter(
                best["gdpval"] * 100,
                best["safety"] * 100,
                c=CARRIED_COLOR,
                s=120,
                marker="*",
                edgecolors="black",
                linewidth=1,
                zorder=6,
            )
            ax.annotate(
                "carried fwd",
                (best["gdpval"] * 100, best["safety"] * 100),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=7,
                color=CARRIED_COLOR,
                style="italic",
            )

    # Plot baselines
    for name, (gdp, safe, marker, color) in baselines.items():
        ax.scatter(
            gdp,
            safe,
            c=color,
            s=70,
            marker=marker,
            edgecolors="black",
            linewidth=0.8,
            zorder=6,
            label=name,
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(25, 85)
    ax.set_title(slice_name, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle="--")


def main():
    base = "results/moevo_balanced_pro"
    output = "docs/paper/figures/pareto_evolution.png"

    programs = load_all_programs(base)
    trajectory = load_trajectory(base)

    if not programs:
        print("No program data found!")
        sys.exit(1)

    print(f"Loaded {len(programs)} programs across {len(set(p['slice'] for p in programs))} slices")

    # Baselines (from results.md, per-slice search-time scores)
    baselines = {
        "Codex CLI": (75.3, 50.5, "s", BASELINE_CODEX),
        "Claude Code": (70.3, 50.4, "^", BASELINE_CLAUDE),
    }

    # Create figure — 2x4 grid for S1-S8
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True, sharey=True)
    fig.suptitle(
        "Pareto Front Evolution Across Development Slices", fontsize=16, fontweight="bold", y=0.98
    )

    for i, s in enumerate(range(1, 9)):
        ax = axes[i // 4][i % 4]
        slice_name = f"S{s}"
        traj_entry = trajectory[i] if i < len(trajectory) else None
        plot_single_slice(ax, programs, traj_entry, slice_name, baselines)

        if i % 4 == 0:
            ax.set_ylabel("ToolEmu Safety (%)")
        if i >= 4:
            ax.set_xlabel("GDPval Task Completion (%)")

    # Legend on last panel
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=SEED_COLOR,
            markersize=8,
            label="Seed program",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#64B5F6",
            markersize=8,
            label="Early iterations",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1565C0",
            markersize=8,
            label="Late iterations",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=FRONT_COLOR,
            markersize=12,
            label="Pareto front",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=CARRIED_COLOR,
            markeredgecolor="black",
            markersize=12,
            label="Carried forward",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=BASELINE_CODEX,
            markeredgecolor="black",
            markersize=8,
            label="Codex CLI baseline",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=BASELINE_CLAUDE,
            markeredgecolor="black",
            markersize=8,
            label="Claude Code baseline",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor="#E0E0E0",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output}")

    # Also save a single-slice detail version for S1 (like AgentBreeder Figure 2)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    [p for p in programs if p["slice"] == "S1"]
    traj_s1 = trajectory[0] if trajectory else None
    plot_single_slice(ax2, programs, traj_s1, "S1", baselines)
    ax2.set_xlabel("GDPval Task Completion (%)")
    ax2.set_ylabel("ToolEmu Safety (%)")
    ax2.set_title("moevo pro: Pareto Front Evolution on S1", fontweight="bold")

    legend_elements_s1 = [
        Line2D(
            [0], [0], marker="D", color="w", markerfacecolor=SEED_COLOR, markersize=8, label="Seed"
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=FRONT_COLOR,
            markersize=12,
            label="Pareto front",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=CARRIED_COLOR,
            markeredgecolor="black",
            markersize=12,
            label="Carried fwd",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=BASELINE_CODEX,
            markeredgecolor="black",
            markersize=8,
            label="Codex CLI",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=BASELINE_CLAUDE,
            markeredgecolor="black",
            markersize=8,
            label="Claude Code",
        ),
    ]
    ax2.legend(
        handles=legend_elements_s1,
        loc="upper left",
        frameon=True,
        fancybox=True,
        edgecolor="#E0E0E0",
    )

    output_s1 = "docs/paper/figures/pareto_evolution_s1.png"
    fig2.savefig(output_s1, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_s1}")

    # Also create the paper figure (single column, 3 representative slices)
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig3.suptitle(
        "Pareto Front Evolution Across Representative Slices",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    for idx, s in enumerate([1, 5, 8]):
        ax = axes3[idx]
        slice_name = f"S{s}"
        traj_entry = trajectory[s - 1] if s - 1 < len(trajectory) else None
        plot_single_slice(ax, programs, traj_entry, slice_name, baselines)
        ax.set_xlabel("GDPval Task Completion (%)")
        if idx == 0:
            ax.set_ylabel("ToolEmu Safety (%)")

    from matplotlib.lines import Line2D

    legend_elements_3 = [
        Line2D(
            [0], [0], marker="D", color="w", markerfacecolor=SEED_COLOR, markersize=8, label="Seed"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#64B5F6",
            markersize=7,
            label="Early iter.",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1565C0",
            markersize=7,
            label="Late iter.",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=FRONT_COLOR,
            markersize=11,
            label="Pareto front",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=CARRIED_COLOR,
            markeredgecolor="black",
            markersize=11,
            label="Carried fwd",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=BASELINE_CODEX,
            markeredgecolor="black",
            markersize=7,
            label="Codex CLI",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=BASELINE_CLAUDE,
            markeredgecolor="black",
            markersize=7,
            label="Claude Code",
        ),
    ]
    fig3.legend(
        handles=legend_elements_3,
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
        fancybox=True,
        edgecolor="#E0E0E0",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    output_3 = "docs/paper/figures/pareto_evolution_3panel.png"
    fig3.savefig(output_3, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_3}")

    # Paper-ready single-column version
    paper_output = "paper/fig_pareto_evolution.png"
    fig3.savefig(paper_output, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {paper_output}")


if __name__ == "__main__":
    main()
