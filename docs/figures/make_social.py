"""Render the GitHub social-preview card (1280x640) and a square icon.

The motif is the mechanism itself: a Pareto staircase of kept candidates in
the capability x safety plane, with the single scalar survivor for contrast.
Upload docs/figures/social_preview.png under Settings > Social preview;
GitHub cannot set it via the API.

Usage:
    python docs/figures/make_social.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

BLUE = "#1a5fb4"
ORANGE = "#e8710a"
INK = "#262626"
MUT = "#666666"
DIM = "#c9cdd3"
BG = "#ffffff"


def draw_motif(ax, lw=3.0, dot=180):
    """The staircase-front glyph on a unit square."""
    front = [(0.18, 0.86), (0.52, 0.66), (0.86, 0.34)]
    # staircase: right, then down
    xs, ys = [front[0][0]], [front[0][1]]
    for i in range(1, len(front)):
        xs += [front[i][0], front[i][0]]
        ys += [front[i - 1][1], front[i][1]]
    xs.append(front[-1][0])
    ys.append(0.16)
    ax.plot(xs, ys, color=BLUE, lw=lw, solid_capstyle="round", zorder=3)
    fx, fy = zip(*front, strict=True)
    ax.scatter(fx, fy, s=dot, color=BLUE, zorder=4, edgecolors=BG, linewidths=lw * 0.7)
    # dominated, kept dim
    dom = [(0.30, 0.44), (0.55, 0.28), (0.40, 0.14)]
    dx, dy = zip(*dom, strict=True)
    ax.scatter(dx, dy, s=dot * 0.55, color=DIM, zorder=2)
    # the scalar survivor, alone
    ax.scatter(
        [0.52],
        [0.66],
        s=dot * 1.9,
        facecolors="none",
        edgecolors=ORANGE,
        linewidths=lw * 0.8,
        zorder=5,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def social() -> None:
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(BG)

    ax_m = fig.add_axes([0.05, 0.20, 0.30, 0.60])
    draw_motif(ax_m, lw=4.4, dot=560)

    x0 = 0.385
    fig.text(x0, 0.600, "MOEvo", fontsize=76, color=INK, fontweight="bold", va="baseline")
    fig.text(
        x0 + 0.003,
        0.480,
        "Multi-objective Pareto evolution of coding-agent harnesses",
        fontsize=18,
        color=MUT,
        va="baseline",
    )

    # colored capability x safety line, placed by measuring each span
    renderer = fig.canvas.get_renderer()
    segs = [
        ("capability", BLUE, "bold"),
        ("  ×  ", MUT, "normal"),
        ("safety", ORANGE, "bold"),
        ("   — no fixed trade-off weight", MUT, "normal"),
    ]
    x = x0 + 0.003
    for txt, col, wt in segs:
        t = fig.text(x, 0.355, txt, fontsize=18, color=col, fontweight=wt, va="baseline")
        bb = t.get_window_extent(renderer=renderer)
        x += bb.width / (fig.get_figwidth() * fig.dpi)

    fig.text(
        x0 + 0.003,
        0.225,
        "NSGA-II  ·  island model  ·  LLM mutation  ·  GDPval + ToolEmu",
        fontsize=17,
        color=MUT,
        va="baseline",
    )

    out = HERE / "social_preview.png"
    fig.savefig(out, dpi=100, facecolor=BG)
    print(f"wrote {out}")


def icon() -> None:
    fig = plt.figure(figsize=(2.56, 2.56), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.10, 0.10, 0.80, 0.80])
    draw_motif(ax, lw=5.0, dot=520)
    out = HERE / "icon.png"
    fig.savefig(out, dpi=200, facecolor=BG)
    print(f"wrote {out}")


if __name__ == "__main__":
    social()
    icon()
