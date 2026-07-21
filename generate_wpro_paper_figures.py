"""Generate publication-style WPRO figures from CSV data.

The script intentionally separates data from rendering:

1. `--create-draft-data` writes reviewable CSV files under `paper_artifacts/figure_data`.
   These draft values are calibrated figure templates, not final publication results.
2. Figure generation always reads CSV files and writes PDF/SVG/PNG outputs.

Replace the CSV files with final held-out test results before paper submission.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.patches import Patch


METHODS = ["FCFS", "EDF", "SRPT", "Utility-Greedy", "Lyapunov", "Vanilla A2C", "PPO", "WPRO"]
METHOD_COLORS = {
    "FCFS": "#8a8a8a",
    "EDF": "#6f6f6f",
    "SRPT": "#525252",
    "Utility-Greedy": "#4c78a8",
    "Lyapunov": "#59a14f",
    "Vanilla A2C": "#b279a2",
    "PPO": "#f28e2b",
    "WPRO": "#d62728",
}
HATCHES = ["", "//", "||", "\\\\", "xx", "..", "++", ""]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_draft_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    overall = [
        ("FCFS", 0.61, 0.030, 0.88, 0.018, 0.52, 0.027),
        ("EDF", 0.70, 0.027, 0.84, 0.020, 0.65, 0.025),
        ("SRPT", 0.66, 0.029, 0.86, 0.018, 0.59, 0.027),
        ("Utility-Greedy", 0.81, 0.023, 0.80, 0.020, 0.73, 0.022),
        ("Lyapunov", 0.78, 0.024, 0.82, 0.018, 0.70, 0.024),
        ("Vanilla A2C", 0.76, 0.031, 0.79, 0.026, 0.68, 0.030),
        ("PPO", 0.80, 0.028, 0.80, 0.024, 0.71, 0.026),
        ("WPRO", 1.00, 0.018, 0.83, 0.016, 0.86, 0.018),
    ]
    write_csv(
        data_dir / "fig1_overall.csv",
        [
            {
                "method": m,
                "normalized_utility_mean": u,
                "normalized_utility_ci": uc,
                "admission_ratio_mean": a,
                "admission_ratio_ci": ac,
                "on_time_ratio_mean": o,
                "on_time_ratio_ci": oc,
            }
            for m, u, uc, a, ac, o, oc in overall
        ],
    )

    rates = [0.16, 0.24, 0.32, 0.40, 0.48]
    rows = []
    curves = {
        "EDF": ([0.79, 0.74, 0.67, 0.55, 0.43], [0.75, 0.70, 0.62, 0.49, 0.35]),
        "Utility-Greedy": ([0.87, 0.84, 0.78, 0.69, 0.58], [0.79, 0.76, 0.71, 0.62, 0.51]),
        "Lyapunov": ([0.84, 0.81, 0.75, 0.66, 0.55], [0.77, 0.74, 0.68, 0.58, 0.48]),
        "Vanilla A2C": ([0.82, 0.78, 0.72, 0.62, 0.50], [0.74, 0.70, 0.64, 0.54, 0.42]),
        "PPO": ([0.84, 0.81, 0.76, 0.67, 0.55], [0.76, 0.73, 0.67, 0.57, 0.46]),
        "WPRO": ([0.91, 0.90, 0.87, 0.81, 0.73], [0.84, 0.84, 0.81, 0.75, 0.67]),
    }
    for method, (utility, ontime) in curves.items():
        for rate, u, o in zip(rates, utility, ontime):
            rows.append(
                {
                    "arrival_rate": rate,
                    "method": method,
                    "normalized_utility_mean": u,
                    "normalized_utility_ci": 0.018 + 0.018 * rate,
                    "on_time_ratio_mean": o,
                    "on_time_ratio_ci": 0.020 + 0.016 * rate,
                    "admission_ratio_mean": min(0.93, 0.88 - 0.18 * max(0.0, rate - 0.16) + (0.02 if method == "WPRO" else 0.0)),
                    "admission_ratio_ci": 0.018,
                }
            )
    write_csv(data_dir / "fig2_scalability.csv", rows)

    envs = [
        ("BurstGPT-low", "Trace", 0.77, 0.84, 520),
        ("BurstGPT-mid", "Trace", 0.73, 0.84, 640),
        ("BurstGPT-heavy", "Trace", 0.66, 0.81, 720),
        ("Azure-like", "Synthetic", 0.71, 0.82, 600),
        ("Tight-SLA", "Synthetic", 0.58, 0.74, 560),
        ("High-load", "Synthetic", 0.61, 0.79, 820),
        ("Cold-load", "Synthetic", 0.63, 0.80, 700),
        ("Hetero-GPU", "Synthetic", 0.67, 0.83, 680),
    ]
    write_csv(
        data_dir / "fig3_robustness.csv",
        [
            {
                "environment_id": e,
                "environment_type": t,
                "best_baseline_utility": b,
                "wpro_utility": w,
                "improvement_pct": 100.0 * (w - b) / b,
                "num_workflows": n,
            }
            for e, t, b, w, n in envs
        ],
    )

    rows = []
    for tightness in [0.75, 0.90, 1.05, 1.20, 1.35]:
        for complexity in [4, 6, 8, 10, 12]:
            baseline = 0.88 - 0.20 * (1.35 - tightness) - 0.018 * (complexity - 4)
            delta = 0.035 + 0.055 * (1.35 - tightness) + 0.010 * (complexity - 4)
            rows.append(
                {
                    "deadline_tightness": tightness,
                    "workflow_complexity": complexity,
                    "best_baseline_utility": baseline,
                    "wpro_utility": min(0.96, baseline + delta),
                    "delta_utility": delta,
                }
            )
    write_csv(data_dir / "fig4_surface.csv", rows)

    breakdown = [
        ("EDF", 17.8, 8.6, 1.8, 25.5),
        ("Utility-Greedy", 14.0, 7.2, 1.7, 25.1),
        ("Lyapunov", 14.8, 6.9, 1.6, 25.4),
        ("Vanilla A2C", 15.5, 7.4, 1.7, 25.2),
        ("PPO", 14.6, 7.0, 1.6, 25.3),
        ("WPRO", 10.5, 4.8, 1.4, 25.8),
    ]
    write_csv(
        data_dir / "fig5_delay_breakdown.csv",
        [
            {"method": m, "queue_waiting": q, "model_preparation": p, "communication": c, "execution": e}
            for m, q, p, c, e in breakdown
        ],
    )

    rng = np.random.default_rng(20260722)
    pred_rows = []
    for model in ["Planner", "Reasoner", "Retriever", "Writer", "Verifier"]:
        for true in np.linspace(0.05, 0.95, 18):
            pred = float(np.clip(true + rng.normal(0.0, 0.055), 0.0, 1.05))
            pred_rows.append({"model": model, "oracle_dag_demand": true, "predicted_demand": pred})
    write_csv(data_dir / "fig6_prediction.csv", pred_rows)
    residency_rows = []
    for load, wpro, greedy in [(0.16, 0.67, 0.58), (0.24, 0.71, 0.60), (0.32, 0.75, 0.62), (0.40, 0.78, 0.61), (0.48, 0.80, 0.59)]:
        for method, hit, loads in [("Utility-Greedy", greedy, 42 - 20 * greedy), ("WPRO", wpro, 42 - 20 * wpro)]:
            residency_rows.append(
                {
                    "arrival_rate": load,
                    "method": method,
                    "residency_hit_mean": hit,
                    "residency_hit_ci": 0.018,
                    "full_loads_mean": loads,
                    "full_loads_ci": 0.9,
                }
            )
    write_csv(data_dir / "fig6_residency.csv", residency_rows)

    ablations = [
        ("WPRO", 1.00, 0.86, 0.78),
        ("WPRO w/o Progress", 0.91, 0.78, 0.74),
        ("WPRO w/o Demand", 0.88, 0.76, 0.68),
        ("WPRO w/o Residency", 0.86, 0.74, 0.61),
        ("WPRO w/o Wait", 0.92, 0.80, 0.72),
        ("WPRO w/o Shaping", 0.94, 0.82, 0.75),
    ]
    write_csv(
        data_dir / "fig7_ablation.csv",
        [
            {
                "variant": v,
                "normalized_utility_mean": u,
                "normalized_utility_ci": 0.020,
                "on_time_ratio_mean": o,
                "on_time_ratio_ci": 0.019,
                "residency_hit_mean": h,
                "residency_hit_ci": 0.018,
            }
            for v, u, o, h in ablations
        ],
    )

    overhead = []
    for candidates in [20, 40, 80, 120, 180, 240]:
        overhead.append({"num_candidates": candidates, "method": "EDF", "decision_ms_mean": 0.08 + 0.0012 * candidates, "decision_ms_ci": 0.012})
        overhead.append({"num_candidates": candidates, "method": "PPO", "decision_ms_mean": 0.30 + 0.0048 * candidates, "decision_ms_ci": 0.035})
        overhead.append({"num_candidates": candidates, "method": "WPRO", "decision_ms_mean": 0.22 + 0.0030 * candidates, "decision_ms_ci": 0.026})
    write_csv(data_dir / "fig8_overhead.csv", overhead)
    timeline = [
        (0, 0.0, 4.6, "Planning", "hit"),
        (1, 0.0, 6.0, "Retrieval", "tool"),
        (2, 0.0, 5.0, "Reasoning", "load"),
        (0, 4.8, 11.2, "Reasoning", "hit"),
        (2, 5.2, 9.5, "Writing", "load"),
        (1, 6.3, 12.4, "Verification", "hit"),
        (2, 9.8, 15.1, "Writing", "hit"),
        (0, 11.5, 16.0, "Repair", "hit"),
        (1, 12.7, 18.3, "Reasoning", "load"),
        (2, 15.4, 20.0, "Verification", "hit"),
    ]
    write_csv(
        data_dir / "fig8_timeline.csv",
        [
            {"gpu": g, "start": s, "end": e, "stage_type": st, "residency": r}
            for g, s, e, st, r in timeline
        ],
    )


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def add_grid(ax) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.55, alpha=0.55)
    ax.set_axisbelow(True)


def fig1_overall(data_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(data_dir / "fig1_overall.csv")
    metrics = [
        ("normalized_utility", r"Normalized utility $U/U_{\mathrm{WPRO}}$"),
        ("admission_ratio", r"Admission ratio $R_{\mathrm{adm}}$"),
        ("on_time_ratio", r"On-time ratio $R_{\mathrm{on}}$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.55), sharex=True)
    x = np.arange(len(df))
    for ax, (metric, ylabel) in zip(axes, metrics):
        vals = df[f"{metric}_mean"].to_numpy()
        ci = df[f"{metric}_ci"].to_numpy()
        bars = ax.bar(x, vals, yerr=ci, capsize=2.5, color=[METHOD_COLORS[m] for m in df["method"]], edgecolor="black", linewidth=0.45)
        for bar, hatch in zip(bars, HATCHES):
            bar.set_hatch(hatch)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.12)
        add_grid(ax)
    axes[1].set_xlabel("Scheduling policy")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df["method"], rotation=35, ha="right")
    handles = [
        Patch(facecolor=METHOD_COLORS[m], edgecolor="black", hatch=HATCHES[i], label=m, linewidth=0.45)
        for i, m in enumerate(df["method"])
    ]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.20), frameon=False)
    save_all(fig, out_dir, "fig1_overall_performance")


def fig2_scalability(data_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(data_dir / "fig2_scalability.csv")
    metrics = [
        ("normalized_utility", r"Normalized utility"),
        ("on_time_ratio", r"On-time ratio"),
        ("admission_ratio", r"Admission ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.55), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for method in ["EDF", "Utility-Greedy", "Lyapunov", "Vanilla A2C", "PPO", "WPRO"]:
            sub = df[df["method"] == method]
            ax.errorbar(
                sub["arrival_rate"],
                sub[f"{metric}_mean"],
                yerr=sub[f"{metric}_ci"],
                marker="*" if method == "WPRO" else "o",
                linewidth=1.5 if method == "WPRO" else 1.0,
                markersize=5 if method == "WPRO" else 3.5,
                color=METHOD_COLORS[method],
                capsize=2.0,
                label=method,
            )
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"Arrival rate $\lambda$")
        ax.set_ylim(0.28, 0.98)
        add_grid(ax)
    axes[0].legend(ncol=2, frameon=False, loc="lower left")
    save_all(fig, out_dir, "fig2_scalability_vs_arrival_rate")


def fig3_robustness(data_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(data_dir / "fig3_robustness.csv")
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    colors = ["#4c78a8" if t == "Trace" else "#59a14f" for t in df["environment_type"]]
    ax.scatter(df["best_baseline_utility"], df["wpro_utility"], s=df["num_workflows"] / 5.0, c=colors, edgecolors="black", linewidths=0.5, alpha=0.82)
    lo = min(df["best_baseline_utility"].min(), df["wpro_utility"].min()) - 0.02
    hi = max(df["best_baseline_utility"].max(), df["wpro_utility"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=0.9, label=r"$y=x$")
    for _, r in df.iterrows():
        ax.text(r["best_baseline_utility"] + 0.004, r["wpro_utility"] + 0.003, str(r["environment_id"]), fontsize=6.5)
    ax.set_xlabel(r"Best baseline utility")
    ax.set_ylabel(r"WPRO utility")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    add_grid(ax)
    ax.legend(frameon=False, loc="upper left")
    save_all(fig, out_dir, "fig3_multi_environment_improvement")


def fig4_surface(data_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(data_dir / "fig4_surface.csv")
    xvals = sorted(df["deadline_tightness"].unique())
    yvals = sorted(df["workflow_complexity"].unique())
    X, Y = np.meshgrid(xvals, yvals)
    Z = df.pivot(index="workflow_complexity", columns="deadline_tightness", values="delta_utility").loc[yvals, xvals].to_numpy()
    fig = plt.figure(figsize=(5.4, 3.8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z * 100.0, cmap=cm.viridis, linewidth=0.25, edgecolor="white", antialiased=True)
    ax.set_xlabel(r"Tightness $\tau$", labelpad=1)
    ax.set_ylabel(r"DAG size $|V_j|$", labelpad=3)
    ax.set_zlabel(r"Gain (\%)", labelpad=2)
    ax.view_init(elev=27, azim=-58)
    fig.colorbar(surf, shrink=0.58, aspect=14, pad=0.13)
    save_all(fig, out_dir, "fig4_deadline_complexity_surface")


def fig5_delay_breakdown(data_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(data_dir / "fig5_delay_breakdown.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    bottom = np.zeros(len(df))
    stacks = [
        ("queue_waiting", "Queue waiting", "#bab0ab"),
        ("model_preparation", "Model preparation", "#f28e2b"),
        ("communication", "Communication", "#76b7b2"),
        ("execution", "Execution", "#4e79a7"),
    ]
    x = np.arange(len(df))
    for col, label, color in stacks:
        ax.bar(x, df[col], bottom=bottom, label=label, color=color, edgecolor="black", linewidth=0.4)
        bottom += df[col].to_numpy()
    ax.set_ylabel(r"Mean completion latency (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=30, ha="right")
    add_grid(ax)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    save_all(fig, out_dir, "fig5_latency_breakdown")


def fig6_mechanism(data_dir: Path, out_dir: Path) -> None:
    pred = pd.read_csv(data_dir / "fig6_prediction.csv")
    res = pd.read_csv(data_dir / "fig6_residency.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axes[0]
    for model, sub in pred.groupby("model"):
        ax.scatter(sub["oracle_dag_demand"], sub["predicted_demand"], s=13, alpha=0.70, label=model)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.9)
    ax.set_xlabel(r"Oracle DAG demand $d_m^{\mathrm{DAG}}(H)$")
    ax.set_ylabel(r"Predicted demand $\hat d_m(H)$")
    add_grid(ax)
    ax.legend(frameon=False, ncol=2, fontsize=6.7)
    ax = axes[1]
    for method, sub in res.groupby("method"):
        ax.errorbar(
            sub["arrival_rate"],
            sub["residency_hit_mean"],
            yerr=sub["residency_hit_ci"],
            marker="*" if method == "WPRO" else "o",
            color=METHOD_COLORS[method],
            linewidth=1.5 if method == "WPRO" else 1.0,
            capsize=2.0,
            label=method,
        )
    ax.set_xlabel(r"Arrival rate $\lambda$")
    ax.set_ylabel(r"Residency hit ratio")
    ax.set_ylim(0.50, 0.86)
    add_grid(ax)
    ax.legend(frameon=False, loc="lower right")
    save_all(fig, out_dir, "fig6_demand_prediction_and_residency")


def fig7_ablation(data_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(data_dir / "fig7_ablation.csv")
    metrics = [
        ("normalized_utility", r"Normalized utility"),
        ("on_time_ratio", r"On-time ratio"),
        ("residency_hit", r"Residency hit ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.6), sharex=True)
    x = np.arange(len(df))
    colors = ["#d62728", "#72b7b2", "#54a24b", "#eeca3b", "#b279a2", "#ff9da6"]
    for ax, (metric, ylabel) in zip(axes, metrics):
        bars = ax.bar(x, df[f"{metric}_mean"], yerr=df[f"{metric}_ci"], capsize=2.3, color=colors, edgecolor="black", linewidth=0.45)
        for bar, hatch in zip(bars, ["", "--", "++", "oo", "\\\\", "**"]):
            bar.set_hatch(hatch)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.55, 1.05)
        add_grid(ax)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df["variant"], rotation=35, ha="right")
    save_all(fig, out_dir, "fig7_ablation_study")


def fig8_behavior(data_dir: Path, out_dir: Path) -> None:
    timeline = pd.read_csv(data_dir / "fig8_timeline.csv")
    overhead = pd.read_csv(data_dir / "fig8_overhead.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    ax = axes[0]
    colors = {"hit": "#59a14f", "load": "#f28e2b", "tool": "#76b7b2"}
    for _, r in timeline.iterrows():
        ax.barh(r["gpu"], r["end"] - r["start"], left=r["start"], color=colors[r["residency"]], edgecolor="black", linewidth=0.4)
        ax.text((r["start"] + r["end"]) / 2, r["gpu"], r["stage_type"], ha="center", va="center", fontsize=6.2)
    ax.set_xlabel(r"Simulation time")
    ax.set_ylabel(r"GPU")
    ax.set_yticks(sorted(timeline["gpu"].unique()))
    ax.invert_yaxis()
    add_grid(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.4) for c in colors.values()]
    ax.legend(handles, ["Resident hit", "Cold load", "Tool/Comm."], frameon=False, fontsize=7, loc="upper right")
    ax = axes[1]
    for method, sub in overhead.groupby("method"):
        ax.errorbar(
            sub["num_candidates"],
            sub["decision_ms_mean"],
            yerr=sub["decision_ms_ci"],
            marker="*" if method == "WPRO" else "o",
            color=METHOD_COLORS[method],
            linewidth=1.5 if method == "WPRO" else 1.0,
            capsize=2.0,
            label=method,
        )
    ax.set_xlabel(r"Candidate actions $|\mathcal{A}_n|$")
    ax.set_ylabel(r"Decision overhead (ms)")
    add_grid(ax)
    ax.legend(frameon=False, loc="upper left")
    save_all(fig, out_dir, "fig8_schedule_timeline_and_overhead")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WPRO paper figures from CSV data.")
    parser.add_argument("--data-dir", type=Path, default=Path("paper_artifacts/figure_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_artifacts/figures"))
    parser.add_argument("--create-draft-data", action="store_true")
    args = parser.parse_args()

    configure_style()
    if args.create_draft_data:
        create_draft_data(args.data_dir)

    fig1_overall(args.data_dir, args.output_dir)
    fig2_scalability(args.data_dir, args.output_dir)
    fig3_robustness(args.data_dir, args.output_dir)
    fig4_surface(args.data_dir, args.output_dir)
    fig5_delay_breakdown(args.data_dir, args.output_dir)
    fig6_mechanism(args.data_dir, args.output_dir)
    fig7_ablation(args.data_dir, args.output_dir)
    fig8_behavior(args.data_dir, args.output_dir)
    print(f"Wrote WPRO paper figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
