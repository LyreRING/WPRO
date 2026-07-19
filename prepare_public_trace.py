"""Prepare public LLM traces for WPRO trace-driven experiments.

The raw BurstGPT file is large and should normally stay local. This helper
downloads the public trace when needed and emits small reproducible subsets for
paper experiments.
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path


BURSTGPT_URL = "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download and subset public BurstGPT traces.")
    p.add_argument("--raw", type=Path, default=Path("data/public_traces/BurstGPT_1.csv"))
    p.add_argument("--output", type=Path, default=Path("data/public_traces/BurstGPT_1_dense_120.csv"))
    p.add_argument("--requests", type=int, default=120)
    p.add_argument("--mode", choices=["dense", "prefix"], default="dense")
    p.add_argument("--download", action="store_true")
    return p


def download_if_needed(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    print(f"Downloading BurstGPT_1.csv to {path} ...")
    urllib.request.urlretrieve(BURSTGPT_URL, path)


def load_clean_rows(path: Path) -> list[tuple[float, dict[str, str]]]:
    rows: list[tuple[float, dict[str, str]]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timestamp = float(row["Timestamp"])
                request_tokens = float(row["Request tokens"])
                response_tokens = float(row["Response tokens"])
            except (KeyError, TypeError, ValueError):
                continue
            if request_tokens <= 0.0 or response_tokens <= 0.0:
                continue
            rows.append((timestamp, row))
    rows.sort(key=lambda item: item[0])
    return rows


def select_rows(rows: list[tuple[float, dict[str, str]]], n: int, mode: str) -> list[dict[str, str]]:
    if mode == "prefix":
        return [row for _, row in rows[:n]]
    best_i = 0
    best_span = float("inf")
    for i in range(0, max(0, len(rows) - n)):
        span = rows[i + n - 1][0] - rows[i][0]
        if span < best_span:
            best_span = span
            best_i = i
    selected = rows[best_i : best_i + n]
    print(
        f"Selected dense window: n={n}, start={selected[0][0]:.3f}, "
        f"end={selected[-1][0]:.3f}, span={best_span:.3f}s"
    )
    return [row for _, row in selected]


def main() -> None:
    args = parser().parse_args()
    if args.download:
        download_if_needed(args.raw)
    rows = load_clean_rows(args.raw)
    selected = select_rows(rows, args.requests, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Timestamp", "Model", "Request tokens", "Response tokens", "Total tokens", "Log Type"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    print(f"Wrote {len(selected)} rows to {args.output}")


if __name__ == "__main__":
    main()
