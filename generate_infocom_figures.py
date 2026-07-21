"""Generate publication-style INFOCOM figures from experiment CSV files.

All figures are generated directly from CSV results. The primary outputs are
vector PDF and SVG files; PNG is emitted only for quick preview.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


POLICY_ORDER = [
    "random",
    "fcfs",
    "edf",
    "srpt",
    "online_greedy",
    "dag_oracle_greedy",
    "vanilla_a2c",
    "wpr_no_progress",
    "wpr_no_demand",
    "wpr_no_residency",
    "wpr_no_shaping",
    "wpr_a2c",
]

POLICY_LABEL = {
    "random": "Random",
    "fcfs": "FCFS",
    "edf": "EDF",
    "srpt": "SRPT",
    "online_greedy": "Online Greedy",
    "dag_oracle_greedy": "DAG-Oracle Greedy",
    "vanilla_a2c": "Vanilla A2C",
    "wpr_no_progress": "WPRO w/o Progress",
    "wpr_no_demand": "WPRO w/o Demand",
    "wpr_no_residency": "WPRO w/o Residency",
    "wpr_no_shaping": "WPRO w/o Shaping",
    "wpr_a2c": "WPRO",
}

POLICY_COLOR = {
    "random": "#9e9e9e",
    "fcfs": "#6f6f6f",
    "edf": "#7f7f7f",
    "srpt": "#525252",
    "online_greedy": "#4c78a8",
    "dag_oracle_greedy": "#f58518",
    "vanilla_a2c": "#b279a2",
    "wpr_no_progress": "#72b7b2",
    "wpr_no_demand": "#54a24b",
    "wpr_no_residency": "#eeca3b",
    "wpr_no_shaping": "#ff9da6",
    "wpr_a2c": "#d62728",
}

POLICY_MARKER = {
    "random": "o",
    "fcfs": "h",
    "edf": "s",
    "srpt": "p",
    "online_greedy": "^",
    "dag_oracle_greedy": "D",
    "vanilla_a2c": "v",
    "wpr_no_progress": "P",
    "wpr_no_demand": "X",
    "wpr_no_residency": "<",
    "wpr_no_shaping": ">",
    "wpr_a2c": "*",
}

POLICY_HATCH = {
    "random": "",
    "fcfs": "..",
    "edf": "//",
    "srpt": "||",
    "online_greedy": "\\\\",
    "dag_oracle_greedy": "xx",
    "vanilla_a2c": "..",
    "wpr_no_progress": "--",
    "wpr_no_demand": "++",
    "wpr_no_residency": "oo",
    "wpr_no_shaping": "**",
    "wpr_a2c": "",
}

METRIC_LABEL = {
    "weighted_completed_value": r"Weighted completed value $V_{\mathrm{w}}$",
    "weighted_goodput_rate": r"Weighted goodput rate $G_{\mathrm{w}}$",
    "sla_success_ratio": r"SLA success ratio",
    "p95_latency": r"P95 latency",
    "avg_ready_wait": r"Average ready waiting time",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate publication-quality figures from WPRO CSV outputs.")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"], choices=["pdf", "svg", "png"])
    p.add_argument("--dpi", type=int, default=300)
    return p


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.8,
            "hatch.linewidth": 0.5,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fval(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def save_figure(fig: plt.Figure, path: Path, formats: list[str], dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, color="0.82", dashes=(3, 2))
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def policy_subset(summary: list[dict[str, str]]) -> list[str]:
    present = {row.get("policy") for row in summary}
    return [p for p in POLICY_ORDER if p in present]


def scenario_order(summary: list[dict[str, str]]) -> list[str]:
    scenarios: list[str] = []
    for row in summary:
        name = row.get("scenario", "trace")
        if name not in scenarios:
            scenarios.append(name)
    return scenarios


def grouped_bar(summary: list[dict[str, str]], metric: str, out: Path, formats: list[str], dpi: int) -> None:
    policies = policy_subset(summary)
    scenarios = scenario_order(summary)
    if not policies or not scenarios:
        return
    lookup = {(r.get("scenario", "trace"), r["policy"]): r for r in summary}
    fig_w = max(3.6, 0.55 * len(policies) * max(1, len(scenarios)))
    fig, ax = plt.subplots(figsize=(fig_w, 2.35))
    x = np.arange(len(scenarios), dtype=float)
    width = min(0.78 / max(1, len(policies)), 0.10)
    offsets = (np.arange(len(policies)) - (len(policies) - 1) / 2.0) * width
    for idx, policy in enumerate(policies):
        means = []
        sems = []
        for scenario in scenarios:
            row = lookup.get((scenario, policy), {})
            means.append(fval(row, f"{metric}_mean"))
            sems.append(fval(row, f"{metric}_sem"))
        ax.bar(
            x + offsets[idx],
            means,
            width=width * 0.92,
            yerr=sems,
            capsize=2.0,
            linewidth=0.45,
            edgecolor="black",
            color=POLICY_COLOR[policy],
            hatch=POLICY_HATCH[policy],
            label=POLICY_LABEL[policy],
            error_kw={"elinewidth": 0.65, "capthick": 0.65},
        )
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in scenarios], rotation=0)
    style_axis(ax)
    ax.legend(
        ncol=min(5, len(policies)),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        frameon=False,
        handlelength=1.25,
        columnspacing=0.8,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    save_figure(fig, out, formats, dpi)


def trace_bar(summary: list[dict[str, str]], out: Path, formats: list[str], dpi: int) -> None:
    policies = policy_subset(summary)
    if not policies:
        return
    lookup = {row["policy"]: row for row in summary}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55))
    for ax, metric in zip(axes, ["weighted_goodput_rate", "sla_success_ratio"]):
        vals = [fval(lookup.get(p, {}), f"{metric}_mean") for p in policies]
        sems = [fval(lookup.get(p, {}), f"{metric}_sem") for p in policies]
        pos = np.arange(len(policies))
        ax.bar(
            pos,
            vals,
            yerr=sems,
            capsize=2,
            color=[POLICY_COLOR[p] for p in policies],
            edgecolor="black",
            linewidth=0.45,
            hatch=[POLICY_HATCH[p] for p in policies],
            error_kw={"elinewidth": 0.65, "capthick": 0.65},
        )
        ax.set_ylabel(METRIC_LABEL[metric])
        ax.set_xticks(pos)
        ax.set_xticklabels([POLICY_LABEL[p] for p in policies], rotation=35, ha="right")
        style_axis(ax)
    save_figure(fig, out, formats, dpi)


def workload_response_line(summary: list[dict[str, str]], out: Path, formats: list[str], dpi: int) -> None:
    scenarios = scenario_order(summary)
    if len(scenarios) < 2:
        return
    policies = [p for p in ["edf", "online_greedy", "vanilla_a2c", "wpr_a2c"] if p in policy_subset(summary)]
    lookup = {(r.get("scenario", "trace"), r["policy"]): r for r in summary}
    x = np.arange(len(scenarios), dtype=float)
    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    for policy in policies:
        y = [fval(lookup.get((s, policy), {}), "weighted_goodput_rate_mean") for s in scenarios]
        ax.plot(
            x,
            y,
            marker=POLICY_MARKER[policy],
            color=POLICY_COLOR[policy],
            label=POLICY_LABEL[policy],
            markeredgecolor="black",
            markeredgewidth=0.35,
        )
    ax.set_xlabel(r"Workload intensity / scenario")
    ax.set_ylabel(METRIC_LABEL["weighted_goodput_rate"])
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in scenarios], rotation=20, ha="right")
    style_axis(ax)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, out, formats, dpi)


def latency_goodput_scatter(summary: list[dict[str, str]], out: Path, formats: list[str], dpi: int) -> None:
    rows = [r for r in summary if "weighted_goodput_rate_mean" in r and "p95_latency_mean" in r]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.65, 2.65))
    for row in rows:
        policy = row["policy"]
        ax.scatter(
            fval(row, "weighted_goodput_rate_mean"),
            fval(row, "p95_latency_mean"),
            s=46 if policy == "wpr_a2c" else 30,
            marker=POLICY_MARKER.get(policy, "o"),
            color=POLICY_COLOR.get(policy, "0.4"),
            edgecolors="black",
            linewidths=0.45,
            label=POLICY_LABEL.get(policy, policy),
            zorder=3,
        )
    ax.set_xlabel(METRIC_LABEL["weighted_goodput_rate"])
    ax.set_ylabel(METRIC_LABEL["p95_latency"])
    style_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="best")
    save_figure(fig, out, formats, dpi)


def ablation_dot(summary: list[dict[str, str]], out: Path, formats: list[str], dpi: int) -> None:
    base_row = next((r for r in summary if r.get("policy") == "wpr_a2c"), None)
    if not base_row:
        return
    base = fval(base_row, "weighted_goodput_rate_mean")
    ablations = ["wpr_no_progress", "wpr_no_demand", "wpr_no_residency", "wpr_no_shaping"]
    rows = [next((r for r in summary if r.get("policy") == p), None) for p in ablations]
    rows = [r for r in rows if r]
    if not rows:
        return
    values = [(fval(r, "weighted_goodput_rate_mean") - base) / max(abs(base), 1e-9) * 100.0 for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(3.55, 2.25))
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.scatter(values, y, s=46, color=[POLICY_COLOR[r["policy"]] for r in rows], edgecolors="black", linewidths=0.45)
    for val, yi in zip(values, y):
        ax.text(val + (1.5 if val >= 0 else -1.5), yi, f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([POLICY_LABEL[r["policy"]] for r in rows])
    ax.set_xlabel(r"Goodput change vs. WPRO (\%)")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, color="0.82", dashes=(3, 2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, out, formats, dpi)


def convergence_curve(training: list[dict[str, str]], out: Path, formats: list[str], dpi: int) -> None:
    policies = [p for p in ["vanilla_a2c", "wpr_no_demand", "wpr_a2c"] if any(r.get("policy") == p for r in training)]
    if not policies:
        return
    fig, ax = plt.subplots(figsize=(3.65, 2.45))
    for policy in policies:
        rows = [r for r in training if r.get("policy") == policy]
        grouped: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            grouped[int(float(r.get("episode", 0.0)))].append(fval(r, "weighted_completed_value"))
        eps = sorted(grouped)
        means = np.asarray([np.mean(grouped[e]) for e in eps], dtype=float)
        if len(means) >= 5:
            kernel = np.ones(5) / 5.0
            means = np.convolve(means, kernel, mode="same")
        ax.plot(eps, means, marker=POLICY_MARKER[policy], markevery=max(1, len(eps) // 6), color=POLICY_COLOR[policy], label=POLICY_LABEL[policy])
    ax.set_xlabel(r"Training episode")
    ax.set_ylabel(r"Training return $V_{\mathrm{w}}$")
    style_axis(ax)
    ax.legend(frameon=False)
    save_figure(fig, out, formats, dpi)


def convergence_surface(training: list[dict[str, str]], out: Path, formats: list[str], dpi: int) -> None:
    policies = [p for p in ["wpr_no_progress", "wpr_no_demand", "wpr_no_residency", "wpr_a2c"] if any(r.get("policy") == p for r in training)]
    if not policies:
        return
    episode_keys = sorted({int(float(r.get("episode", 0.0))) for r in training})
    if not episode_keys:
        return
    z = np.zeros((len(policies), len(episode_keys)), dtype=float)
    lookup_eps = {e: idx for idx, e in enumerate(episode_keys)}
    for pi, policy in enumerate(policies):
        grouped: dict[int, list[float]] = defaultdict(list)
        for r in training:
            if r.get("policy") == policy:
                grouped[int(float(r.get("episode", 0.0)))].append(fval(r, "weighted_completed_value"))
        for e, vals in grouped.items():
            z[pi, lookup_eps[e]] = float(np.mean(vals))
    x, y = np.meshgrid(np.asarray(episode_keys, dtype=float), np.arange(len(policies), dtype=float))
    fig = plt.figure(figsize=(4.35, 3.1))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis", edgecolor="0.35", linewidth=0.15, antialiased=True, alpha=0.94)
    ax.set_xlabel(r"Episode", labelpad=3)
    ax.set_ylabel(r"Variant", labelpad=4)
    ax.set_zlabel(r"$V_{\mathrm{w}}$", labelpad=4)
    ax.set_yticks(np.arange(len(policies)))
    ax.set_yticklabels([POLICY_LABEL[p] for p in policies], fontsize=6)
    ax.view_init(elev=24, azim=-128)
    ax.xaxis._axinfo["grid"]["linestyle"] = "--"
    ax.yaxis._axinfo["grid"]["linestyle"] = "--"
    ax.zaxis._axinfo["grid"]["linestyle"] = "--"
    save_figure(fig, out, formats, dpi)


def parameter_sensitivity(rows: list[dict[str, str]], out: Path, formats: list[str], dpi: int) -> None:
    """Plot parameter trends from parameter_sensitivity.csv.

    Expected schema:
        parameter,value,policy,metric_mean,metric_sem
    """

    if not rows:
        return
    parameters = []
    for row in rows:
        param = row.get("parameter", "parameter")
        if param not in parameters:
            parameters.append(param)
    for param in parameters:
        sub = [r for r in rows if r.get("parameter", "parameter") == param]
        policies = [p for p in POLICY_ORDER if any(r.get("policy") == p for r in sub)]
        if not policies:
            continue
        fig, ax = plt.subplots(figsize=(3.55, 2.35))
        for policy in policies:
            pr = sorted([r for r in sub if r.get("policy") == policy], key=lambda r: fval(r, "value"))
            x = np.asarray([fval(r, "value") for r in pr], dtype=float)
            y = np.asarray([fval(r, "metric_mean") for r in pr], dtype=float)
            e = np.asarray([fval(r, "metric_sem") for r in pr], dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=e,
                marker=POLICY_MARKER.get(policy, "o"),
                color=POLICY_COLOR.get(policy, "0.4"),
                label=POLICY_LABEL.get(policy, policy),
                capsize=2,
                elinewidth=0.65,
                markeredgecolor="black",
                markeredgewidth=0.35,
            )
        ax.set_xlabel(param.replace("_", " "))
        ax.set_ylabel(r"Weighted goodput rate $G_{\mathrm{w}}$")
        style_axis(ax)
        ax.legend(frameon=False, ncol=2)
        save_figure(fig, out.with_name(f"{out.name}_{param}"), formats, dpi)


def parameter_surface(rows: list[dict[str, str]], out: Path, formats: list[str], dpi: int) -> None:
    """Plot a 3D parameter surface when two-parameter sensitivity data exists.

    Expected schema:
        x_parameter,x_value,y_parameter,y_value,metric_mean
    """

    if not rows or not {"x_value", "y_value", "metric_mean"}.issubset(rows[0]):
        return
    xs = sorted({fval(r, "x_value") for r in rows})
    ys = sorted({fval(r, "y_value") for r in rows})
    if len(xs) < 2 or len(ys) < 2:
        return
    z = np.full((len(ys), len(xs)), np.nan, dtype=float)
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    for r in rows:
        z[yi[fval(r, "y_value")], xi[fval(r, "x_value")]] = fval(r, "metric_mean")
    if np.isnan(z).any():
        return
    x, y = np.meshgrid(np.asarray(xs), np.asarray(ys))
    fig = plt.figure(figsize=(3.95, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis", edgecolor="0.35", linewidth=0.15, antialiased=True)
    ax.set_xlabel(rows[0].get("x_parameter", "x"), labelpad=3)
    ax.set_ylabel(rows[0].get("y_parameter", "y"), labelpad=3)
    ax.set_zlabel(r"$G_{\mathrm{w}}$", labelpad=4)
    ax.view_init(elev=24, azim=-132)
    ax.xaxis._axinfo["grid"]["linestyle"] = "--"
    ax.yaxis._axinfo["grid"]["linestyle"] = "--"
    ax.zaxis._axinfo["grid"]["linestyle"] = "--"
    save_figure(fig, out, formats, dpi)


def main() -> None:
    args = parser().parse_args()
    configure_matplotlib()
    out = args.output_dir or (args.input_dir / "paper_figures")
    summary = read_csv(args.input_dir / "summary_metrics.csv")
    training = read_csv(args.input_dir / "training_curve.csv")
    sensitivity = read_csv(args.input_dir / "parameter_sensitivity.csv")
    sensitivity_2d = read_csv(args.input_dir / "parameter_surface.csv")

    if summary:
        grouped_bar(summary, "weighted_completed_value", out / "fig_grouped_bar_weighted_value", args.formats, args.dpi)
        grouped_bar(summary, "weighted_goodput_rate", out / "fig_grouped_bar_goodput_rate", args.formats, args.dpi)
        grouped_bar(summary, "sla_success_ratio", out / "fig_grouped_bar_sla", args.formats, args.dpi)
        trace_bar(summary, out / "fig_trace_driven_bar", args.formats, args.dpi)
        workload_response_line(summary, out / "fig_workload_response_line", args.formats, args.dpi)
        latency_goodput_scatter(summary, out / "fig_latency_goodput_pareto", args.formats, args.dpi)
        ablation_dot(summary, out / "fig_ablation_dot_plot", args.formats, args.dpi)
    if training:
        convergence_curve(training, out / "fig_convergence_curve_2d", args.formats, args.dpi)
        convergence_surface(training, out / "fig_convergence_surface_3d", args.formats, args.dpi)
    parameter_sensitivity(sensitivity, out / "fig_parameter_sensitivity", args.formats, args.dpi)
    parameter_surface(sensitivity_2d, out / "fig_parameter_surface_3d", args.formats, args.dpi)
    print(f"Wrote publication-style figures to {out.resolve()}")


if __name__ == "__main__":
    main()
