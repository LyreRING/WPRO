"""Trace-driven WPR-A2C experiment entry.

This script uses production-derived request traces for the arrival process and
token lengths, then maps each real request to an application workflow template:

    real LLM request trace + agentic application template
    = trace-driven workflow instance.

Synthetic workloads remain useful for controlled sensitivity studies; this
entry is intended for realistic evaluation once BurstGPT/Azure/Mooncake-style
CSV traces are available.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dag_a2c.wpr_a2c import WPRA2CConfig, train_wpr_agent
from dag_a2c.wpr_baselines import (
    dag_oracle_residency_greedy,
    edf_matching,
    online_ready_greedy,
    random_matching,
)
from dag_a2c.wpr_env import TraceWorkloadSource, WPREnv
from run_wpr_experiments import POLICY_SEED_OFFSET, write_csv


POLICIES = [
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
    p = argparse.ArgumentParser(description="Run trace-driven WPR-A2C experiments.")
    p.add_argument("--trace-path", type=Path, default=None, help="Backward-compatible single CSV trace. Use train/validation/test trace paths for final paper runs.")
    p.add_argument("--train-trace-path", type=Path, default=None, help="Training split CSV trace.")
    p.add_argument("--validation-trace-path", type=Path, default=None, help="Validation split CSV trace for checkpoint selection.")
    p.add_argument("--test-trace-path", type=Path, default=None, help="Held-out test split CSV trace for final evaluation.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=20260719)
    p.add_argument("--horizon", type=float, default=80.0)
    p.add_argument("--max-active", type=int, default=8)
    p.add_argument("--arrival-rate", type=float, default=0.30, help="Used only by env defaults; trace timestamps drive arrivals.")
    p.add_argument("--time-scale", type=float, default=1.0, help="Compress trace time by this factor.")
    p.add_argument("--max-requests", type=int, default=None)
    p.add_argument("--duration", type=float, default=None, help="Keep trace requests whose scaled arrival time is within this window.")
    p.add_argument("--deadline-mode", choices=["template", "relative", "elapsed"], default="relative")
    p.add_argument("--deadline-multiplier", type=float, default=2.5)
    p.add_argument("--checkpoint-metric", choices=["weighted_completed_value", "weighted_goodput_rate", "sla_success_ratio"], default="weighted_goodput_rate")
    p.add_argument("--validation-interval", type=int, default=5)
    p.add_argument("--validation-episodes", type=int, default=1)
    p.add_argument("--timestamp-col", default=None)
    p.add_argument("--input-tokens-col", default=None)
    p.add_argument("--output-tokens-col", default=None)
    p.add_argument("--model-col", default=None)
    p.add_argument("--elapsed-col", default=None)
    p.add_argument("--quick", action="store_true")
    return p


def resolve_trace_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    train = args.train_trace_path or args.trace_path
    validation = args.validation_trace_path or args.trace_path
    test = args.test_trace_path or args.trace_path
    missing = [
        name
        for name, value in (("train", train), ("validation", validation), ("test", test))
        if value is None
    ]
    if missing:
        raise ValueError(
            "Missing trace path(s): "
            + ", ".join(missing)
            + ". Use --trace-path for smoke runs or --train-trace-path/--validation-trace-path/--test-trace-path for paper runs."
        )
    return Path(train), Path(validation), Path(test)


def make_source(args: argparse.Namespace, trace_path: Path) -> TraceWorkloadSource:
    return TraceWorkloadSource(
        path=trace_path,
        time_scale=args.time_scale,
        max_requests=args.max_requests,
        duration=args.duration,
        timestamp_col=args.timestamp_col,
        input_tokens_col=args.input_tokens_col,
        output_tokens_col=args.output_tokens_col,
        model_col=args.model_col,
        elapsed_col=args.elapsed_col,
        deadline_mode=args.deadline_mode,
        deadline_multiplier=args.deadline_multiplier,
    )


def make_env(args: argparse.Namespace, seed: int, trace_path: Path) -> WPREnv:
    return WPREnv(
        horizon=args.horizon,
        arrival_rate=args.arrival_rate,
        max_active=args.max_active,
        seed=seed,
        workload_source=make_source(args, trace_path),
    )


def eval_policy(args: argparse.Namespace, seed: int, policy_name: str, policy, trace_path: Path) -> dict[str, float | str | int]:
    env = make_env(args, seed, trace_path)
    env.reset(seed)
    while not env.done:
        env.step(policy(env))
    return {"scenario": "trace_driven", "policy": policy_name, "seed": seed, **env.final_metrics()}


def eval_agent(args: argparse.Namespace, seed: int, agent, policy_name: str, trace_path: Path) -> dict[str, float | str | int]:
    env = make_env(args, seed, trace_path)
    env.reset(seed)
    while not env.done:
        assn, _ = agent.dispatch(env, deterministic=True)
        env.step(assn)
    return {"scenario": "trace_driven", "policy": policy_name, "seed": seed, **env.final_metrics()}


def summarize(rows: list[dict]) -> list[dict[str, float | str]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["policy"]].append(row)
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
    )
    summary = []
    for policy in POLICIES:
        vals = groups.get(policy, [])
        if not vals:
            continue
        row: dict[str, float | str] = {"scenario": "trace_driven", "policy": policy, "n": float(len(vals))}
        for metric in metrics:
            arr = np.asarray([float(v.get(metric, 0.0)) for v in vals], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(arr))
            row[f"{metric}_sem"] = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        summary.append(row)
    return summary


def _font(size=20, bold=False):
    for p in [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _color(policy: str) -> str:
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


def draw_trace_bars(path: Path, summary: list[dict], metric: str, title: str) -> None:
    values = [float(r[f"{metric}_mean"]) for r in summary]
    vmax = max(values + [1e-9])
    img = Image.new("RGB", (1500, 900), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 35), title, fill="#17202A", font=_font(34, True))
    d.text((55, 82), "Trace-driven workload: real arrival timestamps and token lengths mapped to workflow DAG templates.", fill="#5B6572", font=_font(19))
    x0, y0, w, h = 90, 170, 1320, 510
    d.line((x0, y0, x0, y0 + h), fill="#17202A", width=2)
    d.line((x0, y0 + h, x0 + w, y0 + h), fill="#17202A", width=2)
    for tick in range(6):
        val = vmax * tick / 5
        yy = y0 + h - val / max(vmax, 1e-9) * h
        d.line((x0 - 8, yy, x0 + w, yy), fill="#E7EBF0", width=1)
        d.text((x0 - 12, yy), f"{val:.2f}" if vmax < 2 else f"{val:.1f}", fill="#5B6572", font=_font(14), anchor="rm")
    step = w / max(1, len(summary))
    bar_w = min(72, max(32, int(step * 0.55)))
    for i, row in enumerate(summary):
        policy = str(row["policy"])
        value = float(row[f"{metric}_mean"])
        sem = float(row[f"{metric}_sem"])
        x = x0 + i * step + (step - bar_w) / 2
        y = y0 + h - value / max(vmax, 1e-9) * h
        d.rounded_rectangle((x, y, x + bar_w, y0 + h), radius=5, fill=_color(policy))
        err = sem / max(vmax, 1e-9) * h
        d.line((x + bar_w / 2, y - err, x + bar_w / 2, y + err), fill="#17202A", width=1)
        label = policy.replace("_", "\n")
        d.multiline_text((x + bar_w / 2, y0 + h + 22), label, fill="#17202A", font=_font(14, True), anchor="ma", align="center", spacing=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)


def draw_trace_characterization(path: Path, trace_path: Path) -> None:
    with trace_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    def norm(name: str) -> str:
        return name.strip().lower().replace(" ", "_").replace("-", "_")

    keys = {norm(k): k for k in rows[0]}
    ts_key = keys.get("timestamp") or keys.get("time") or keys.get("arrival")
    in_key = keys.get("request_tokens") or keys.get("input_tokens") or keys.get("prompt_tokens")
    out_key = keys.get("response_tokens") or keys.get("output_tokens") or keys.get("completion_tokens")
    if ts_key is None or in_key is None or out_key is None:
        return
    ts = np.asarray([float(r[ts_key]) for r in rows], dtype=float)
    ins = np.asarray([float(r[in_key]) for r in rows], dtype=float)
    outs = np.asarray([float(r[out_key]) for r in rows], dtype=float)
    gaps = np.diff(np.sort(ts)) if len(ts) > 1 else np.asarray([0.0])
    img = Image.new("RGB", (1400, 820), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 35), "Trace workload characterization", fill="#17202A", font=_font(34, True))
    d.text((55, 82), f"requests={len(rows)}, input tokens mean={np.mean(ins):.1f}, output tokens mean={np.mean(outs):.1f}", fill="#5B6572", font=_font(19))
    panels = [(ins, "input tokens", "#2A60B0", 80), (outs, "output tokens", "#60B6A4", 500), (gaps, "inter-arrival gap", "#DA9A37", 920)]
    for arr, title, col, x0 in panels:
        y0, w, h = 170, 350, 460
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=8, fill="#F7F9FC", outline="#D7DEE9", width=2)
        d.text((x0 + 20, y0 + 18), title, fill="#17202A", font=_font(22, True))
        hist, bins = np.histogram(arr, bins=min(12, max(4, len(arr) // 2)))
        ymax = max(int(hist.max()), 1)
        bw = (w - 70) / len(hist)
        for i, cnt in enumerate(hist):
            bh = cnt / ymax * (h - 115)
            x = x0 + 42 + i * bw
            y = y0 + h - 40 - bh
            d.rounded_rectangle((x, y, x + bw * 0.72, y0 + h - 40), radius=3, fill=col)
        d.text((x0 + 20, y0 + h - 28), f"min={arr.min():.1f}, p50={np.median(arr):.1f}, max={arr.max():.1f}", fill="#5B6572", font=_font(15))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)


def main() -> None:
    args = parser().parse_args()
    train_trace_path, validation_trace_path, test_trace_path = resolve_trace_paths(args)
    if args.quick:
        args.episodes = 6
        args.eval_episodes = 2
        args.seeds = 1
        args.max_requests = args.max_requests or 18
        args.horizon = min(args.horizon, 45.0)
    if args.output is None:
        args.output = Path("outputs/wpr_trace_quick") if args.quick else Path("outputs/wpr_trace_full")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_trace_path": str(train_trace_path),
                "validation_trace_path": str(validation_trace_path),
                "test_trace_path": str(test_trace_path),
                "uses_isolated_trace_splits": len({str(train_trace_path), str(validation_trace_path), str(test_trace_path)}) == 3,
                "checkpoint_metric": args.checkpoint_metric,
                "validation_interval": args.validation_interval,
                "validation_episodes": args.validation_episodes,
                "episodes": args.episodes,
                "eval_episodes": args.eval_episodes,
                "seeds": args.seeds,
            },
            f,
            indent=2,
        )

    baselines = {
        "random": random_matching,
        "edf": edf_matching,
        "online_greedy": online_ready_greedy,
        "dag_oracle_greedy": dag_oracle_residency_greedy,
    }
    agents_cfg = {
        "vanilla_a2c": WPRA2CConfig(
            use_progress_encoder=False,
            use_demand_predictor=False,
            use_residency_scorer=False,
            use_residency_features=False,
            use_wait_features=False,
            allow_wait=False,
            structural_prior_strength=0.0,
        ),
        "wpr_no_progress": WPRA2CConfig(use_progress_encoder=False),
        "wpr_no_demand": WPRA2CConfig(use_demand_predictor=False),
        "wpr_no_residency": WPRA2CConfig(use_residency_scorer=False, use_residency_features=False),
        "wpr_no_shaping": WPRA2CConfig(use_potential_shaping=False),
        "wpr_a2c": WPRA2CConfig(actor_lr=0.003, critic_lr=0.008, entropy_coef=0.0005, allow_wait=False, use_wait_features=False),
    }
    for cfg in agents_cfg.values():
        cfg.checkpoint_metric = args.checkpoint_metric
        cfg.validation_interval = args.validation_interval
        cfg.validation_episodes = args.validation_episodes

    rows: list[dict] = []
    train_rows: list[dict] = []
    for seed_idx in range(args.seeds):
        train_seed = args.base_seed + 10000 * seed_idx
        print(f"[trace-driven] train/eval seed={train_seed}")
        agents = {}
        for name, cfg in agents_cfg.items():
            cfg.seed = train_seed + POLICY_SEED_OFFSET[name]
            agent, curve = train_wpr_agent(
                lambda sd, a=args, p=train_trace_path: make_env(a, sd, p),
                args.episodes,
                cfg.seed,
                cfg,
                validation_env_factory=lambda sd, a=args, p=validation_trace_path: make_env(a, sd, p),
            )
            agents[name] = agent
            for r in curve:
                train_rows.append({"scenario": "trace_driven", "policy": name, "train_seed": train_seed, **r})
        for ep in range(args.eval_episodes):
            eval_seed = args.base_seed + 50000 + 10000 * seed_idx + 100 * ep
            for pname, policy in baselines.items():
                rows.append(eval_policy(args, eval_seed, pname, policy, test_trace_path))
            for pname, agent in agents.items():
                rows.append(eval_agent(args, eval_seed, agent, pname, test_trace_path))
        summary = summarize(rows)
        write_csv(out / "episode_metrics.csv", rows)
        write_csv(out / "training_curve.csv", train_rows)
        write_csv(out / "summary_metrics.csv", summary)

    summary = summarize(rows)
    write_csv(out / "episode_metrics.csv", rows)
    write_csv(out / "training_curve.csv", train_rows)
    write_csv(out / "summary_metrics.csv", summary)
    draw_trace_bars(out / "figures" / "weighted_completed_value.png", summary, "weighted_completed_value", "Trace-driven weighted completed value")
    draw_trace_bars(out / "figures" / "weighted_goodput_rate.png", summary, "weighted_goodput_rate", "Trace-driven weighted goodput rate")
    draw_trace_bars(out / "figures" / "sla_success_ratio.png", summary, "sla_success_ratio", "Trace-driven SLA success ratio")
    draw_trace_bars(out / "figures" / "p95_latency.png", summary, "p95_latency", "Trace-driven P95 latency")
    draw_trace_characterization(out / "figures" / "trace_characterization.png", test_trace_path)
    print(f"Wrote trace-driven WPR-A2C experiments to {out.resolve()}")


if __name__ == "__main__":
    main()
