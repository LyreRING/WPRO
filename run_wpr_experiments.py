"""Complete WPR-A2C experiment entry.

本脚本只从真实 CSV 实验结果生成图，不手工绘制结果。工程修正：
- --quick 不再覆盖用户显式指定的 --output；
- policy seed 使用固定 offset，避免 Python hash 随进程随机化；
- vanilla_a2c 是真正训练得到的普通 A2C 配置，而不是启发式；
- weighted_goodput 拆成 weighted_completed_value 与 weighted_goodput_rate。
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dag_a2c.wpr_a2c import WPRA2CConfig, train_wpr_agent
from dag_a2c.wpr_baselines import (
    dag_oracle_residency_greedy,
    edf_matching,
    lookahead_search_upper_reference,
    online_ready_greedy,
    random_matching,
)
from dag_a2c.wpr_env import WPREnv


@dataclass(frozen=True)
class Scenario:
    name: str
    horizon: float
    arrival_rate: float
    max_active: int


SCENARIOS = (
    Scenario("small_reference", 18.0, 0.16, 3),
    Scenario("light", 45.0, 0.18, 5),
    Scenario("moderate", 55.0, 0.28, 7),
    Scenario("heavy", 65.0, 0.38, 8),
)

POLICY_SEED_OFFSET = {
    "vanilla_a2c": 11,
    "wpr_no_progress": 101,
    "wpr_no_demand": 211,
    "wpr_no_residency": 307,
    "wpr_fixed_gamma": 409,
    "wpr_no_shaping": 463,
    "wpr_a2c": 521,
}

POLICIES = [
    "random",
    "edf",
    "online_greedy",
    "dag_oracle_greedy",
    "vanilla_a2c",
    "wpr_no_progress",
    "wpr_no_demand",
    "wpr_no_residency",
    "wpr_fixed_gamma",
    "wpr_no_shaping",
    "wpr_a2c",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run WPR-A2C workflow-progress/residency experiments.")
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=20260717)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--quick", action="store_true")
    return p


def make_env(s: Scenario, seed: int) -> WPREnv:
    return WPREnv(horizon=s.horizon, arrival_rate=s.arrival_rate, max_active=s.max_active, seed=seed)


def eval_policy(s: Scenario, seed: int, policy_name: str, policy) -> dict[str, float | str | int]:
    env = make_env(s, seed)
    env.reset(seed)
    while not env.done:
        env.step(policy(env))
    return {"scenario": s.name, "policy": policy_name, "seed": seed, **env.final_metrics()}


def eval_agent(s: Scenario, seed: int, agent, policy_name: str) -> dict[str, float | str | int]:
    env = make_env(s, seed)
    env.reset(seed)
    while not env.done:
        assn, _ = agent.dispatch(env, deterministic=True)
        env.step(assn)
    return {"scenario": s.name, "policy": policy_name, "seed": seed, **env.final_metrics()}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict[str, float | str]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["policy"])].append(row)
    metrics = (
        "weighted_completed_value",
        "weighted_goodput_rate",
        "weighted_goodput_ratio",
        "sla_success_ratio",
        "completion_ratio",
        "p95_latency",
        "avg_latency",
        "avg_ready_wait",
        "rejected",
        "dropped",
        "lookahead_gap",
    )
    out = []
    for (scenario, policy), vals in sorted(groups.items()):
        row: dict[str, float | str] = {"scenario": scenario, "policy": policy, "n": float(len(vals))}
        for metric in metrics:
            arr = np.asarray([float(v.get(metric, 0.0)) for v in vals], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(arr))
            row[f"{metric}_sem"] = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        out.append(row)
    return out


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
        "wpr_fixed_gamma": "#E08D67",
        "wpr_no_shaping": "#5B8A72",
        "wpr_a2c": "#2A60B0",
    }.get(policy, "#677076")


def draw_grouped_bars(path: Path, summary: list[dict], metric: str, title: str, higher: bool = True) -> None:
    scenarios = [s.name for s in SCENARIOS]
    lookup = {(r["scenario"], r["policy"]): r for r in summary}
    vals = [float(lookup[(s, p)][f"{metric}_mean"]) for s in scenarios for p in POLICIES if (s, p) in lookup]
    vmax = max(vals + [1e-9])
    img = Image.new("RGB", (1900, 1040), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 34), title, fill="#17202A", font=font(34, True))
    d.text((60, 82), f"Metric: {metric}; {'higher' if higher else 'lower'} is better. Error bars show SEM.", fill="#5B6572", font=font(20))
    x0, y0, w, h = 90, 175, 1720, 590
    d.line((x0, y0, x0, y0 + h), fill="#17202A", width=2)
    d.line((x0, y0 + h, x0 + w, y0 + h), fill="#17202A", width=2)
    for tick in range(6):
        val = vmax * tick / 5
        yy = y0 + h - (val / max(vmax, 1e-9)) * h
        d.line((x0 - 8, yy, x0 + w, yy), fill="#E7EBF0", width=1)
        d.text((x0 - 14, yy), f"{val:.2f}" if vmax < 2 else f"{val:.1f}", fill="#5B6572", font=font(14), anchor="rm")
    group_w = w / len(scenarios)
    bar_w = 20
    for si, scenario in enumerate(scenarios):
        gx = x0 + si * group_w + 42
        for pi, policy in enumerate(POLICIES):
            row = lookup.get((scenario, policy))
            if not row:
                continue
            v = float(row[f"{metric}_mean"])
            sem = float(row[f"{metric}_sem"])
            bh = v / max(vmax, 1e-9) * h
            x = gx + pi * (bar_w + 9)
            y = y0 + h - bh
            d.rounded_rectangle((x, y, x + bar_w, y0 + h), radius=4, fill=color(policy))
            err = sem / max(vmax, 1e-9) * h
            d.line((x + bar_w / 2, y - err, x + bar_w / 2, y + err), fill="#17202A", width=1)
        d.text((gx + 160, y0 + h + 38), scenario, fill="#17202A", font=font(20, True), anchor="mm")
    lx, ly = 95, 840
    for i, policy in enumerate(POLICIES):
        xx = lx + (i % 5) * 360
        yy = ly + (i // 5) * 48
        d.rounded_rectangle((xx, yy, xx + 28, yy + 18), radius=4, fill=color(policy))
        d.text((xx + 40, yy - 4), policy, fill="#17202A", font=font(18, True))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)


def draw_training(path: Path, rows: list[dict]) -> None:
    img = Image.new("RGB", (1500, 860), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 34), "WPR-A2C training diagnostics", fill="#17202A", font=font(34, True))
    d.text((60, 82), "Curves are generated from actual training_curve.csv.", fill="#5B6572", font=font(20))
    if not rows:
        img.save(path)
        return
    wpr_rows = [r for r in rows if r.get("policy") == "wpr_a2c"]
    if not wpr_rows:
        wpr_rows = rows
    eps = np.asarray([float(r["episode"]) for r in wpr_rows])
    value = np.asarray([float(r["weighted_completed_value"]) for r in wpr_rows])
    loss = np.asarray([float(r["demand_loss"]) for r in wpr_rows])
    panels = [(value, "weighted completed value", "#2A60B0", 120), (loss, "demand prediction loss", "#C94B45", 780)]
    for arr, title, col, x0 in panels:
        y0, w, h = 170, 560, 480
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=8, fill="#F7F9FC", outline="#D7DEE9", width=2)
        d.text((x0 + 24, y0 + 20), title, fill="#17202A", font=font(22, True))
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi - lo < 1e-9:
            hi = lo + 1.0
        pts = []
        for i, val in enumerate(arr):
            x = x0 + 60 + (eps[i] - eps.min()) / max(1.0, eps.max() - eps.min()) * (w - 100)
            y = y0 + 80 + (hi - val) / (hi - lo) * (h - 140)
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            d.line((a[0], a[1], b[0], b[1]), fill=col, width=4)
    img.save(path, quality=95)


def main() -> None:
    args = parser().parse_args()
    scenarios = list(SCENARIOS)
    output_was_default = args.output is None
    if args.quick:
        args.episodes = 8
        args.eval_episodes = 2
        args.seeds = 1
        scenarios = [SCENARIOS[1]]
    if args.output is None:
        args.output = Path("outputs/wpr_a2c_quick") if args.quick else Path("outputs/wpr_a2c_full")
    out = args.output
    if args.quick and not output_was_default:
        print(f"[quick] preserving user output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    train_rows: list[dict] = []
    baselines = {
        "random": random_matching,
        "edf": edf_matching,
        "online_greedy": online_ready_greedy,
        "dag_oracle_greedy": dag_oracle_residency_greedy,
    }
    agents_cfg = {
        "vanilla_a2c": WPRA2CConfig(use_progress_encoder=False, use_demand_predictor=False, use_residency_scorer=False),
        "wpr_no_progress": WPRA2CConfig(use_progress_encoder=False),
        "wpr_no_demand": WPRA2CConfig(use_demand_predictor=False),
        "wpr_no_residency": WPRA2CConfig(use_residency_scorer=False),
        "wpr_fixed_gamma": WPRA2CConfig(use_time_critic=False),
        "wpr_no_shaping": WPRA2CConfig(use_potential_shaping=False),
        "wpr_a2c": WPRA2CConfig(),
    }

    for sidx, scenario in enumerate(scenarios):
        for seed_idx in range(args.seeds):
            train_seed = args.base_seed + 10000 * seed_idx + 100 * sidx
            print(f"[scenario={scenario.name}] train/eval seed={train_seed}")
            agents = {}
            for name, cfg in agents_cfg.items():
                agent_seed = train_seed + POLICY_SEED_OFFSET[name]
                cfg.seed = agent_seed
                agent, curve = train_wpr_agent(lambda sd, sc=scenario: make_env(sc, sd), args.episodes, agent_seed, cfg)
                agents[name] = agent
                for r in curve:
                    train_rows.append({"scenario": scenario.name, "policy": name, "train_seed": train_seed, **r})
            for ep in range(args.eval_episodes):
                eval_seed = args.base_seed + 50000 + 10000 * seed_idx + 100 * ep + 7 * sidx
                reference = lookahead_search_upper_reference(eval_seed, horizon=scenario.horizon, arrival_rate=scenario.arrival_rate) if scenario.name == "small_reference" else 0.0
                for pname, policy in baselines.items():
                    row = eval_policy(scenario, eval_seed, pname, policy)
                    row["lookahead_reference_value"] = reference
                    row["lookahead_gap"] = (reference - row["weighted_completed_value"]) / max(reference, 1e-9) if reference > 0 else 0.0
                    rows.append(row)
                for pname, agent in agents.items():
                    row = eval_agent(scenario, eval_seed, agent, pname)
                    row["lookahead_reference_value"] = reference
                    row["lookahead_gap"] = (reference - row["weighted_completed_value"]) / max(reference, 1e-9) if reference > 0 else 0.0
                    rows.append(row)

    summary = summarize(rows)
    write_csv(out / "episode_metrics.csv", rows)
    write_csv(out / "training_curve.csv", train_rows)
    write_csv(out / "summary_metrics.csv", summary)
    draw_grouped_bars(out / "figures" / "weighted_completed_value.png", summary, "weighted_completed_value", "Weighted completed value", higher=True)
    draw_grouped_bars(out / "figures" / "weighted_goodput_rate.png", summary, "weighted_goodput_rate", "Weighted goodput rate", higher=True)
    draw_grouped_bars(out / "figures" / "sla_success_ratio.png", summary, "sla_success_ratio", "SLA success ratio", higher=True)
    draw_grouped_bars(out / "figures" / "p95_latency.png", summary, "p95_latency", "P95 latency under event-driven serving", higher=False)
    draw_grouped_bars(out / "figures" / "lookahead_gap.png", summary, "lookahead_gap", "Small-scale lookahead gap", higher=False)
    draw_training(out / "figures" / "wpr_training_diagnostics.png", train_rows)
    print(f"Wrote WPR-A2C experiments to {out.resolve()}")


if __name__ == "__main__":
    main()
