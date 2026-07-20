"""Generate INFOCOM-style figures from experiment CSV outputs.

The script reads existing CSV result files and emits both PNG and PDF versions.
It deliberately avoids hand-entered values: every figure is derived from
summary_metrics.csv, episode_metrics.csv, or training_curve.csv.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


POLICY_ORDER = [
    "random",
    "edf",
    "online_greedy",
    "dag_oracle_greedy",
    "vanilla_a2c",
    "wpr_no_progress",
    "wpr_no_demand",
    "wpr_no_residency",
    "wpr_no_shaping",
    "wpr_a2c",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate figures from WPRO experiment CSV files.")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    return p


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def font(size=20, bold=False):
    for p in [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def color(policy: str) -> str:
    return {
        "random": "#C94B45",
        "edf": "#8A7C2F",
        "online_greedy": "#DA9A37",
        "dag_oracle_greedy": "#B47C2B",
        "vanilla_a2c": "#6D6F8C",
        "wpr_no_progress": "#78A6C8",
        "wpr_no_demand": "#60B6A4",
        "wpr_no_residency": "#9B84C6",
        "wpr_no_shaping": "#5B8A72",
        "wpr_a2c": "#2A60B0",
    }.get(policy, "#677076")


def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path.with_suffix(".png"), quality=95)
    img.save(path.with_suffix(".pdf"), "PDF", resolution=150)


def grouped_bar(summary: list[dict[str, str]], metric: str, title: str, out: Path) -> None:
    rows = [r for r in summary if f"{metric}_mean" in r]
    scenarios = []
    for r in rows:
        if r.get("scenario", "trace_driven") not in scenarios:
            scenarios.append(r.get("scenario", "trace_driven"))
    lookup = {(r.get("scenario", "trace_driven"), r["policy"]): float(r[f"{metric}_mean"]) for r in rows}
    vals = list(lookup.values()) or [1.0]
    vmax = max(vals + [1e-9])
    img = Image.new("RGB", (1800, 980), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 34), title, fill="#17202A", font=font(34, True))
    d.text((60, 82), f"Metric: {metric}. Bars and error bars are computed from CSV results.", fill="#5B6572", font=font(19))
    x0, y0, w, h = 90, 170, 1620, 560
    d.line((x0, y0, x0, y0 + h), fill="#17202A", width=2)
    d.line((x0, y0 + h, x0 + w, y0 + h), fill="#17202A", width=2)
    for tick in range(6):
        val = vmax * tick / 5
        yy = y0 + h - val / max(vmax, 1e-9) * h
        d.line((x0 - 6, yy, x0 + w, yy), fill="#E7EBF0")
        d.text((x0 - 12, yy), f"{val:.2f}" if vmax < 2 else f"{val:.1f}", fill="#5B6572", font=font(14), anchor="rm")
    group_w = w / max(1, len(scenarios))
    bar_w = max(12, min(26, int(group_w / (len(POLICY_ORDER) + 5))))
    for si, scenario in enumerate(scenarios):
        gx = x0 + si * group_w + 35
        for pi, policy in enumerate(POLICY_ORDER):
            key = (scenario, policy)
            if key not in lookup:
                continue
            val = lookup[key]
            sem = float(next((r.get(f"{metric}_sem", 0.0) for r in rows if r.get("scenario", "trace_driven") == scenario and r["policy"] == policy), 0.0))
            bh = val / max(vmax, 1e-9) * h
            x = gx + pi * (bar_w + 8)
            y = y0 + h - bh
            d.rounded_rectangle((x, y, x + bar_w, y0 + h), radius=3, fill=color(policy))
            err = sem / max(vmax, 1e-9) * h
            d.line((x + bar_w / 2, y - err, x + bar_w / 2, y + err), fill="#17202A")
        d.text((gx + group_w * 0.36, y0 + h + 36), scenario, fill="#17202A", font=font(18, True), anchor="mm")
    legend_y = 820
    for i, policy in enumerate(POLICY_ORDER):
        x = 70 + (i % 5) * 335
        y = legend_y + (i // 5) * 45
        d.rounded_rectangle((x, y, x + 26, y + 16), radius=3, fill=color(policy))
        d.text((x + 38, y - 4), policy, fill="#17202A", font=font(16, True))
    save(img, out)


def scatter_latency_goodput(summary: list[dict[str, str]], out: Path) -> None:
    pts = []
    for r in summary:
        if "weighted_goodput_rate_mean" in r and "p95_latency_mean" in r:
            pts.append((r["policy"], float(r["weighted_goodput_rate_mean"]), float(r["p95_latency_mean"])))
    img = Image.new("RGB", (1200, 860), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 35), "Latency-goodput Pareto view", fill="#17202A", font=font(32, True))
    x0, y0, w, h = 110, 140, 950, 560
    xmax = max([p[1] for p in pts] + [1.0])
    ymax = max([p[2] for p in pts] + [1.0])
    ymin = min([p[2] for p in pts] + [0.0])
    d.rectangle((x0, y0, x0 + w, y0 + h), outline="#17202A", width=2)
    for policy, goodput, latency in pts:
        x = x0 + goodput / max(xmax, 1e-9) * w
        y = y0 + (ymax - latency) / max(ymax - ymin, 1e-9) * h
        d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color(policy), outline="#17202A")
        d.text((x + 12, y - 10), policy, fill="#17202A", font=font(13, True))
    d.text((x0 + w / 2, y0 + h + 45), "weighted goodput rate (higher is better)", fill="#17202A", font=font(18), anchor="mm")
    d.text((34, y0 + h / 2), "P95 latency (lower is better)", fill="#17202A", font=font(18), anchor="lm")
    save(img, out)


def ablation_dot(summary: list[dict[str, str]], out: Path) -> None:
    base = next((float(r["weighted_goodput_rate_mean"]) for r in summary if r.get("policy") == "wpr_a2c"), 0.0)
    rows = [r for r in summary if r.get("policy", "").startswith("wpr_")]
    img = Image.new("RGB", (1300, 820), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 35), "Ablation impact", fill="#17202A", font=font(32, True))
    x0, y0, w, gap = 420, 135, 720, 58
    for i, r in enumerate(rows):
        policy = r["policy"]
        val = float(r.get("weighted_goodput_rate_mean", 0.0))
        rel = (val - base) / max(abs(base), 1e-9)
        y = y0 + i * gap
        d.text((80, y - 11), policy, fill="#17202A", font=font(17, True))
        d.line((x0, y, x0 + w, y), fill="#D7DEE9", width=2)
        cx = x0 + (rel + 1.0) / 2.0 * w
        d.ellipse((cx - 8, y - 8, cx + 8, y + 8), fill=color(policy), outline="#17202A")
        d.text((x0 + w + 25, y - 11), f"{rel * 100:+.1f}%", fill="#17202A", font=font(16))
    d.line((x0 + w / 2, y0 - 25, x0 + w / 2, y0 + len(rows) * gap), fill="#17202A", width=2)
    d.text((x0 + w / 2, y0 + len(rows) * gap + 25), "relative goodput change vs WPR-A2C", fill="#17202A", font=font(18), anchor="mm")
    save(img, out)


def convergence(training: list[dict[str, str]], out: Path) -> None:
    rows = [r for r in training if r.get("policy") == "wpr_a2c"]
    if not rows:
        rows = training
    img = Image.new("RGB", (1350, 820), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 35), "2D convergence curve", fill="#17202A", font=font(32, True))
    x0, y0, w, h = 110, 140, 1080, 540
    vals = np.asarray([float(r.get("weighted_completed_value", 0.0)) for r in rows], dtype=float)
    eps = np.arange(len(vals), dtype=float)
    ymax = max(float(vals.max()) if len(vals) else 1.0, 1.0)
    d.rectangle((x0, y0, x0 + w, y0 + h), outline="#17202A", width=2)
    if len(vals) > 1:
        pts = []
        for e, v in zip(eps, vals):
            x = x0 + e / max(eps.max(), 1.0) * w
            y = y0 + (ymax - v) / max(ymax, 1e-9) * h
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            d.line((a[0], a[1], b[0], b[1]), fill="#2A60B0", width=4)
    save(img, out)


def surface(training: list[dict[str, str]], out: Path) -> None:
    rows = [r for r in training if r.get("policy", "").startswith("wpr")]
    img = Image.new("RGB", (1350, 860), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 35), "3D-style convergence surface", fill="#17202A", font=font(32, True))
    groups = defaultdict(list)
    for r in rows:
        groups[r["policy"]].append(float(r.get("weighted_completed_value", 0.0)))
    policies = list(groups)
    max_len = max([len(v) for v in groups.values()] + [1])
    vmax = max([max(v) for v in groups.values() if v] + [1.0])
    x0, y0 = 120, 170
    cell_w, cell_h = 22, 28
    for py, policy in enumerate(policies):
        d.text((x0 - 20, y0 + py * cell_h + 3), policy, fill="#17202A", font=font(12), anchor="ra")
        for ex, val in enumerate(groups[policy]):
            ratio = val / max(vmax, 1e-9)
            shade = int(245 - 150 * ratio)
            col = (shade, int(225 - 80 * ratio), int(255 - 20 * ratio))
            x = x0 + ex * cell_w
            y = y0 + py * cell_h
            d.rectangle((x, y, x + cell_w - 2, y + cell_h - 2), fill=col)
    d.text((x0 + max_len * cell_w / 2, y0 + len(policies) * cell_h + 38), "episode", fill="#17202A", font=font(18), anchor="mm")
    save(img, out)


def workload_response_line(episodes: list[dict[str, str]], out: Path) -> None:
    groups = defaultdict(list)
    for r in episodes:
        groups[r["policy"]].append(float(r.get("weighted_goodput_rate", 0.0)))
    img = Image.new("RGB", (1300, 820), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 35), "Workload-response line chart", fill="#17202A", font=font(32, True))
    x0, y0, w, h = 95, 135, 1040, 540
    vmax = max([max(v) for v in groups.values() if v] + [1.0])
    d.rectangle((x0, y0, x0 + w, y0 + h), outline="#17202A", width=2)
    for policy in POLICY_ORDER:
        vals = groups.get(policy, [])
        if len(vals) < 2:
            continue
        pts = []
        for i, val in enumerate(vals):
            x = x0 + i / max(len(vals) - 1, 1) * w
            y = y0 + (vmax - val) / max(vmax, 1e-9) * h
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            d.line((a[0], a[1], b[0], b[1]), fill=color(policy), width=3)
    save(img, out)


def main() -> None:
    args = parser().parse_args()
    out = args.output_dir or (args.input_dir / "infocom_figures")
    summary = read_csv(args.input_dir / "summary_metrics.csv")
    episodes = read_csv(args.input_dir / "episode_metrics.csv")
    training = read_csv(args.input_dir / "training_curve.csv")
    if summary:
        grouped_bar(summary, "weighted_completed_value", "Grouped weighted completed value", out / "grouped_bar_weighted_value")
        grouped_bar(summary, "weighted_goodput_rate", "Trace-driven weighted goodput rate", out / "trace_driven_bar_goodput")
        scatter_latency_goodput(summary, out / "latency_goodput_pareto")
        ablation_dot(summary, out / "ablation_dot_plot")
    if episodes:
        workload_response_line(episodes, out / "workload_response_line")
    if training:
        convergence(training, out / "convergence_curve_2d")
        surface(training, out / "convergence_surface_3d")
    print(f"Wrote figures to {out.resolve()}")


if __name__ == "__main__":
    main()
