"""Generate split WPRO experiment panels for LaTeX subfigure layout.

The main paper should compose multi-panel figures in LaTeX rather than using
one pre-composed bitmap/PDF. This script reads the same CSV figure data and
emits one clean vector panel per metric/mechanism.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_artifacts" / "figure_data"
OUT = ROOT / "paper" / "Figures" / "split"

METHOD_COLORS = {
    "FCFS": "#8a8a8a",
    "EDF": "#6f6f6f",
    "SRPT": "#525252",
    "Utility-Greedy": "#4c78a8",
    "Lyapunov": "#59a14f",
    "Vanilla A2C": "#b279a2",
    "PPO": "#f28e2b",
    "WPRO": "#d62728",
    "WPRO w/o Progress": "#72b7b2",
    "WPRO w/o Demand": "#54a24b",
    "WPRO w/o Residency": "#eeca3b",
    "WPRO w/o Wait": "#b279a2",
    "WPRO w/o Shaping": "#ff9da6",
}
HATCH = {
    "FCFS": "",
    "EDF": "//",
    "SRPT": "||",
    "Utility-Greedy": "\\\\",
    "Lyapunov": "xx",
    "Vanilla A2C": "..",
    "PPO": "++",
    "WPRO": "",
    "WPRO w/o Progress": "--",
    "WPRO w/o Demand": "++",
    "WPRO w/o Residency": "oo",
    "WPRO w/o Wait": "\\\\",
    "WPRO w/o Shaping": "**",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=320)
    plt.close(fig)


def grid(ax) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def bar_panel(df: pd.DataFrame, metric: str, ylabel: str, name: str, ylim=(0, 1.08), legend=False) -> None:
    fig, ax = plt.subplots(figsize=(3.05, 1.95))
    x = np.arange(len(df))
    bars = ax.bar(
        x,
        df[f"{metric}_mean"],
        yerr=df[f"{metric}_ci"],
        capsize=2.0,
        color=[METHOD_COLORS[m] for m in df["method"]],
        edgecolor="black",
        linewidth=0.42,
    )
    for b, m in zip(bars, df["method"]):
        b.set_hatch(HATCH.get(m, ""))
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=35, ha="right")
    grid(ax)
    if legend:
        handles = [Patch(facecolor=METHOD_COLORS[m], edgecolor="black", hatch=HATCH.get(m, ""), label=m) for m in df["method"]]
        ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.42), frameon=False)
    save(fig, name)


def line_panel(df: pd.DataFrame, metric: str, ylabel: str, name: str, ylim=(0.25, 0.98), legend=False) -> None:
    fig, ax = plt.subplots(figsize=(3.05, 1.95))
    for method, sub in df.groupby("method", sort=False):
        if method not in ["EDF", "Utility-Greedy", "Lyapunov", "Vanilla A2C", "PPO", "WPRO"]:
            continue
        ax.errorbar(
            sub["arrival_rate"],
            sub[f"{metric}_mean"],
            yerr=sub[f"{metric}_ci"],
            marker="*" if method == "WPRO" else "o",
            markersize=4.5 if method == "WPRO" else 3.0,
            linewidth=1.35 if method == "WPRO" else 0.95,
            color=METHOD_COLORS[method],
            capsize=1.8,
            label=method,
        )
    ax.set_xlabel(r"Arrival rate $\lambda$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    grid(ax)
    if legend:
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.30), frameon=False, columnspacing=0.8)
    save(fig, name)


def overall() -> None:
    df = pd.read_csv(DATA / "fig1_overall.csv")
    bar_panel(df, "normalized_utility", r"$U/U_{\mathrm{WPRO}}$", "fig_overall_utility", legend=True)
    bar_panel(df, "admission_ratio", r"$R_{\mathrm{adm}}$", "fig_overall_admission")
    bar_panel(df, "on_time_ratio", r"$R_{\mathrm{on}}$", "fig_overall_ontime")


def scalability() -> None:
    df = pd.read_csv(DATA / "fig2_scalability.csv")
    line_panel(df, "normalized_utility", "Normalized utility", "fig_scale_utility", legend=True)
    line_panel(df, "on_time_ratio", r"$R_{\mathrm{on}}$", "fig_scale_ontime")
    line_panel(df, "admission_ratio", r"$R_{\mathrm{adm}}$", "fig_scale_admission", ylim=(0.65, 0.96))


def robustness() -> None:
    df = pd.read_csv(DATA / "fig3_robustness.csv")
    fig, ax = plt.subplots(figsize=(3.1, 2.05))
    colors = ["#4c78a8" if t == "Trace" else "#59a14f" for t in df["environment_type"]]
    ax.scatter(df["best_baseline_utility"], df["wpro_utility"], s=df["num_workflows"] / 6.5, c=colors, edgecolors="black", linewidths=0.45, alpha=0.84)
    lo = min(df["best_baseline_utility"].min(), df["wpro_utility"].min()) - 0.015
    hi = max(df["best_baseline_utility"].max(), df["wpro_utility"].max()) + 0.015
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=0.8, label=r"$y=x$")
    ax.set_xlabel("Best baseline utility")
    ax.set_ylabel("WPRO utility")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    grid(ax)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig_robustness_scatter")


def latency_breakdown() -> None:
    df = pd.read_csv(DATA / "fig5_delay_breakdown.csv")
    fig, ax = plt.subplots(figsize=(3.15, 2.05))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    stacks = [
        ("queue_waiting", "Queue", "#bab0ab"),
        ("model_preparation", "Prep.", "#f28e2b"),
        ("communication", "Comm.", "#76b7b2"),
        ("execution", "Exec.", "#4e79a7"),
    ]
    for col, label, color in stacks:
        ax.bar(x, df[col], bottom=bottom, label=label, color=color, edgecolor="black", linewidth=0.35)
        bottom += df[col].to_numpy()
    ax.set_ylabel("Mean latency (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=30, ha="right")
    grid(ax)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, columnspacing=0.8)
    save(fig, "fig_latency_breakdown_split")


def mechanism() -> None:
    pred = pd.read_csv(DATA / "fig6_prediction.csv")
    fig, ax = plt.subplots(figsize=(3.05, 2.05))
    for model, sub in pred.groupby("model"):
        ax.scatter(sub["oracle_dag_demand"], sub["predicted_demand"], s=10, alpha=0.70, label=model)
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=0.8)
    ax.set_xlabel(r"$d_m^{\mathrm{DAG}}(H)$")
    ax.set_ylabel(r"$\hat d_m(H)$")
    grid(ax)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.30), frameon=False, columnspacing=0.8)
    save(fig, "fig_mech_prediction")

    res = pd.read_csv(DATA / "fig6_residency.csv")
    fig, ax = plt.subplots(figsize=(3.05, 2.05))
    for method, sub in res.groupby("method"):
        ax.errorbar(
            sub["arrival_rate"],
            sub["residency_hit_mean"],
            yerr=sub["residency_hit_ci"],
            marker="*" if method == "WPRO" else "o",
            color=METHOD_COLORS[method],
            linewidth=1.35 if method == "WPRO" else 0.95,
            capsize=1.8,
            label=method,
        )
    ax.set_xlabel(r"Arrival rate $\lambda$")
    ax.set_ylabel("Residency hit ratio")
    ax.set_ylim(0.52, 0.84)
    grid(ax)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, borderpad=0.2)
    save(fig, "fig_mech_residency")


def ablation() -> None:
    df = pd.read_csv(DATA / "fig7_ablation.csv").rename(columns={"variant": "method"})
    for metric, ylabel, name in [
        ("normalized_utility", "Normalized utility", "fig_ablation_utility"),
        ("on_time_ratio", r"$R_{\mathrm{on}}$", "fig_ablation_ontime"),
        ("residency_hit", "Residency hit ratio", "fig_ablation_residency"),
    ]:
        bar_panel(df, metric, ylabel, name, ylim=(0.55, 1.05), legend=(metric == "normalized_utility"))


def behavior() -> None:
    timeline = pd.read_csv(DATA / "fig8_timeline.csv")
    fig, ax = plt.subplots(figsize=(3.1, 2.0))
    colors = {"hit": "#59a14f", "load": "#f28e2b", "tool": "#76b7b2"}
    labels = {"hit": "Resident hit", "load": "Cold load", "tool": "Tool/Comm."}
    for _, r in timeline.iterrows():
        ax.barh(r["gpu"], r["end"] - r["start"], left=r["start"], color=colors[r["residency"]], edgecolor="black", linewidth=0.35)
    ax.set_xlabel("Simulation time")
    ax.set_ylabel("GPU")
    ax.set_yticks(sorted(timeline["gpu"].unique()))
    ax.invert_yaxis()
    grid(ax)
    handles = [Patch(facecolor=colors[k], edgecolor="black", label=labels[k]) for k in ["hit", "load", "tool"]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3, frameon=False)
    save(fig, "fig_behavior_timeline")

    over = pd.read_csv(DATA / "fig8_overhead.csv")
    fig, ax = plt.subplots(figsize=(3.05, 2.0))
    for method, sub in over.groupby("method"):
        ax.errorbar(
            sub["num_candidates"],
            sub["decision_ms_mean"],
            yerr=sub["decision_ms_ci"],
            marker="*" if method == "WPRO" else "o",
            color=METHOD_COLORS[method],
            linewidth=1.35 if method == "WPRO" else 0.95,
            capsize=1.8,
            label=method,
        )
    ax.set_xlabel(r"Candidate actions $|\mathcal{A}_k|$")
    ax.set_ylabel("Decision overhead (ms)")
    grid(ax)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, borderpad=0.2)
    save(fig, "fig_behavior_overhead")


def main() -> None:
    style()
    overall()
    scalability()
    robustness()
    latency_breakdown()
    mechanism()
    ablation()
    behavior()
    print(f"Wrote split figures to {OUT}")


if __name__ == "__main__":
    main()
