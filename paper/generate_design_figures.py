"""Generate compact vector design figures for the WPRO paper draft."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def box(ax, xy, w, h, text, fc="#f8f8f8", ec="#222222", fs=8, lw=0.9):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)
    return patch


def arrow(ax, start, end, text=None, rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=0.9,
        color="#222222",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.025, text, ha="center", va="bottom", fontsize=7)


def main() -> None:
    setup()
    out = Path("paper/Figures")
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.03, 0.66), 0.18, 0.22, "Workflow Users\nSLA, quality, value", "#eaf2ff")
    box(ax, (0.29, 0.66), 0.23, 0.22, "AIaaS Orchestrator\nadmission + dispatch", "#edf8ef")
    box(ax, (0.61, 0.66), 0.33, 0.22, "Heterogeneous GPU Cluster\nresident / preparing / running", "#fff3df")
    arrow(ax, (0.21, 0.77), (0.29, 0.77), "workflow arrivals")
    arrow(ax, (0.52, 0.77), (0.61, 0.77), "stage-model-GPU actions")
    arrow(ax, (0.75, 0.66), (0.43, 0.58), "events", rad=-0.20)

    box(ax, (0.05, 0.24), 0.16, 0.12, "Plan", "#f7f7f7")
    box(ax, (0.25, 0.24), 0.16, 0.12, "Retrieve", "#f7f7f7")
    box(ax, (0.45, 0.24), 0.16, 0.12, "Reason", "#f7f7f7")
    box(ax, (0.65, 0.24), 0.16, 0.12, "Write", "#f7f7f7")
    box(ax, (0.82, 0.24), 0.13, 0.12, "Verify", "#f7f7f7")
    for s, e in [((0.21, 0.30), (0.25, 0.30)), ((0.41, 0.30), (0.45, 0.30)), ((0.61, 0.30), (0.65, 0.30)), ((0.81, 0.30), (0.82, 0.30))]:
        arrow(ax, s, e)
    ax.text(0.50, 0.17, "workflow progress releases future stages", ha="center", va="center", fontsize=8)

    box(ax, (0.22, 0.02), 0.56, 0.12, "WPRO: progress encoder + DAG-induced demand estimator + residency-aware actor + time-aware critic", "#f3ecff", fs=7.5)
    arrow(ax, (0.50, 0.24), (0.50, 0.14), "future model demand")
    arrow(ax, (0.78, 0.66), (0.68, 0.14), "model residency", rad=0.25)
    ax.text(0.50, 0.48, "Future coupling: scheduling determines not only what runs next,\n but also which models will be needed and resident next.", ha="center", va="center", fontsize=9, fontweight="bold")

    fig.savefig(out / "fig_system_future_coupling.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_system_future_coupling.png", bbox_inches="tight", dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    main()
