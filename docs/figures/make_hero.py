"""Render the README hero animation: Pareto vs scalar selection, side by side.

Both panels replay the same stream of candidate harnesses in the objective
plane (GDPval capability x ToolEmu safety). The left panel keeps every
non-dominated candidate -- the NSGA-II front advances as a staircase and
dominated points stay in the population, dimmed. The right panel keeps only
the argmax of 0.5*g + 0.5*s -- one survivor at a time, everything else
discarded.

Front anchor points and per-slice seed/best scores come from the real S1-S8
run data (hero_data.json, extracted from the project site). The dominated
candidate cloud between those anchors is synthesized with a fixed seed so the
animation is reproducible; it illustrates the mechanism rather than plotting
per-iteration logs, which were not retained per candidate.

Usage:
    python docs/figures/make_hero.py          # writes hero.gif next to itself
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

HERE = Path(__file__).parent

# Palette: validated blue/orange pair on white; gray is the recessive layer.
BLUE = "#1a5fb4"
ORANGE = "#e8710a"
INK = "#262626"
MUT = "#666666"
DIM = "#c9cdd3"
GRID = "#eceef0"
BG = "#ffffff"

FPS = 18
W, H = 1000, 500
DPI = 100


def pareto_front(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated subset (maximize both), sorted by x."""
    out = []
    for p in pts:
        if not any(q[0] >= p[0] and q[1] >= p[1] and q != p for q in pts):
            out.append(p)
    return sorted(set(out))


def synth_candidates(rng: np.random.Generator, sl: dict) -> list[tuple[float, float]]:
    """Candidate stream for one slice: real front anchors + dominated fill."""
    front = [(f["g"], f["s"]) for f in sl["front"]]
    seed = (max(0.05, sl["seed"]), 0.45 + 0.1 * rng.random())
    gmax = max(g for g, _ in front)
    smax = max(s for _, s in front)
    fills = []
    # exploration cloud: mutations scatter well below and left of the front --
    # regressions included. Their being kept (dimmed) on the Pareto side and
    # discarded on the scalar side is the mechanism the animation exists to show.
    lo_g = max(0.12, min(seed[0], gmax) - 0.45)
    for _ in range(8):
        g = lo_g + (gmax - lo_g) * rng.beta(1.9, 1.1)
        s = 0.30 + (smax - 0.30 + 0.04) * rng.beta(1.7, 1.5)
        g = min(g, gmax - 0.02)
        s = min(s, 0.78)
        fills.append((round(g, 3), round(s, 3)))
    order = fills + front
    rng.shuffle(order)
    # seed first, then the shuffled arrivals, front points never first
    return [seed] + order


def staircase(front: list[tuple[float, float]]):
    """Step path through a maximizing front, for drawing."""
    f = sorted(front)
    xs, ys = [], []
    for i, (x, y) in enumerate(f):
        if i:
            xs.append(x)
            ys.append(f[i - 1][1])
        xs.append(x)
        ys.append(y)
    return xs, ys


def draw_frame(ax_l, ax_r, state, slice_label, t_flash):
    for ax, title, col in (
        (ax_l, "NSGA-II Pareto selection  (MOEvo)", BLUE),
        (ax_r, "scalar selection  ($w{=}0.5$)", ORANGE),
    ):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0.25, 0.8)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.3, 0.45, 0.6, 0.75])
        ax.set_xticklabels(["0", "", "50", "", "100"], fontsize=9, color=MUT)
        ax.set_yticklabels(["30", "45", "60", "75"], fontsize=9, color=MUT)
        ax.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(DIM)
        ax.set_title(title, fontsize=12.5, color=col, pad=10, fontweight="bold")
        ax.set_xlabel("capability  ·  GDPval %", fontsize=10, color=MUT)
    ax_l.set_ylabel("safety  ·  ToolEmu %", fontsize=10, color=MUT)

    pts = state["pts"]
    if not pts:
        return

    # ---- left: Pareto ----
    front = pareto_front(pts)
    dominated = [p for p in pts if p not in front]
    if dominated:
        xs, ys = zip(*dominated, strict=True)
        ax_l.scatter(xs, ys, s=42, color=DIM, zorder=2)
    if len(front) > 1:
        sx, sy = staircase(front)
        # extend the staircase to the plot edges so the dominated region reads
        sx = [sx[0], *sx, sx[-1]]
        sy = [sy[0], *sy, 0.25]
        ax_l.plot(sx, sy, color=BLUE, lw=2, zorder=3, alpha=0.85)
    fx, fy = zip(*front, strict=True)
    ax_l.scatter(fx, fy, s=86, color=BLUE, zorder=4, edgecolors=BG, linewidths=1.4)
    ax_l.annotate(
        f"front: {len(front)} kept · population: {len(pts)}",
        (0.03, 0.965),
        xycoords="axes fraction",
        fontsize=10.5,
        color=BLUE,
        va="top",
    )

    # ---- right: scalar ----
    best = max(pts, key=lambda p: 0.5 * p[0] + 0.5 * p[1])
    losers = [p for p in pts if p != best]
    if losers:
        xs, ys = zip(*losers, strict=True)
        ax_r.scatter(xs, ys, s=42, facecolors="none", edgecolors=DIM, linewidths=1.0, zorder=2)
    c = 0.5 * best[0] + 0.5 * best[1]
    gx = np.array([0.0, 1.0])
    ax_r.plot(gx, 2 * c - gx, color=ORANGE, lw=1.2, ls=(0, (4, 3)), alpha=0.65, zorder=3)
    ax_r.scatter([best[0]], [best[1]], s=100, color=ORANGE, zorder=4, edgecolors=BG, linewidths=1.4)
    ax_r.annotate(
        f"kept: 1 · discarded: {len(losers)}",
        (0.03, 0.965),
        xycoords="axes fraction",
        fontsize=10.5,
        color=ORANGE,
        va="top",
    )

    # newest arrival flash on both panels
    if state.get("new") and t_flash > 0:
        nx, ny = state["new"]
        for ax in (ax_l, ax_r):
            ax.scatter(
                [nx],
                [ny],
                s=200 * t_flash,
                facecolors="none",
                edgecolors=INK,
                linewidths=1.2,
                zorder=5,
            )


def main() -> None:
    slices = json.loads((HERE / "hero_data.json").read_text())
    rng = np.random.default_rng(7)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.115, wspace=0.16)

    frames: list[Image.Image] = []

    def snap(hold=1):
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        img = Image.fromarray(buf)
        frames.extend([img] * hold)

    header: dict = {}

    def chip(text):
        # slice indicator, drawn once in figure coords and updated per slice
        if not header:
            header["title"] = fig.text(
                0.5,
                0.955,
                "one candidate stream  ·  two selection rules",
                ha="center",
                fontsize=13.5,
                color=INK,
                fontweight="bold",
            )
            header["box"] = FancyBboxPatch(
                (0.468, 0.855),
                0.064,
                0.062,
                transform=fig.transFigure,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                fc="#f2f4f6",
                ec=DIM,
                lw=1,
            )
            fig.patches.append(header["box"])
            header["chip"] = fig.text(
                0.5,
                0.885,
                text,
                ha="center",
                va="center",
                fontsize=12,
                color=INK,
                fontweight="bold",
            )
        header["chip"].set_text(text)

    for si, sl in enumerate(slices):
        arrivals = synth_candidates(rng, sl)
        state: dict = {"pts": [], "new": None}
        chip(sl["slice"])
        for _ai, p in enumerate(arrivals):
            state["pts"].append(p)
            state["new"] = p
            for t in (1.0, 0.55):
                draw_frame(ax_l, ax_r, state, sl["slice"], t)
                snap()
            draw_frame(ax_l, ax_r, state, sl["slice"], 0)
            snap(2)
        # hold on the completed slice
        draw_frame(ax_l, ax_r, state, sl["slice"], 0)
        snap(10 if si < len(slices) - 1 else 16)

    # end card
    header["title"].set_text("after 8 slices")
    header["box"].set_visible(False)
    header["chip"].set_text("")
    fig.text(
        0.5,
        0.885,
        "GDPval dev average:  MOEvo 82.0%   ·   scalar 61.7%  (with 2× the iterations)",
        ha="center",
        fontsize=12,
        color=MUT,
    )
    snap(46)

    out = HERE / "hero.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"wrote {out} — {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
