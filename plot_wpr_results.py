"""Regenerate WPR-A2C figures from CSV outputs.

用法：
    py plot_wpr_results.py --input outputs/wpr_a2c_full

本脚本只读取已有 CSV，不重新训练算法，保证论文/PPT 图可以追溯到实验数据。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from run_wpr_experiments import draw_grouped_bars, draw_training


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot WPR-A2C results from summary/training CSV files.")
    parser.add_argument("--input", type=Path, default=Path("outputs/wpr_a2c_full"))
    args = parser.parse_args()

    summary = read_csv(args.input / "summary_metrics.csv")
    training_path = args.input / "training_curve.csv"
    training = read_csv(training_path) if training_path.exists() else []
    fig = args.input / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    metric_names = {key for row in summary for key in row}
    value_metric = "weighted_completed_value" if "weighted_completed_value_mean" in metric_names else "weighted_goodput"
    rate_metric = "weighted_goodput_rate" if "weighted_goodput_rate_mean" in metric_names else value_metric
    gap_metric = "lookahead_gap" if "lookahead_gap_mean" in metric_names else "optimality_gap"

    draw_grouped_bars(fig / f"{value_metric}.png", summary, value_metric, "Weighted completed value", higher=True)
    draw_grouped_bars(fig / f"{rate_metric}.png", summary, rate_metric, "Weighted goodput rate", higher=True)
    draw_grouped_bars(fig / "sla_success_ratio.png", summary, "sla_success_ratio", "SLA success ratio", higher=True)
    draw_grouped_bars(fig / "p95_latency.png", summary, "p95_latency", "P95 latency under event-driven serving", higher=False)
    draw_grouped_bars(fig / f"{gap_metric}.png", summary, gap_metric, "Small-scale lookahead gap", higher=False)
    draw_training(fig / "wpr_training_diagnostics.png", training)
    print(f"Regenerated WPR-A2C figures in {fig.resolve()}")


if __name__ == "__main__":
    main()
