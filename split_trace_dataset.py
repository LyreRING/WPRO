"""Time-ordered train/validation/test split for public LLM traces.

The split is intentionally chronological rather than random. A guard interval
around split boundaries is removed by default, preventing workflows near one
boundary from leaking temporal structure into the next split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Split a trace CSV into chronological train/validation/test splits.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--timestamp-col", default=None)
    p.add_argument("--input-tokens-col", default=None)
    p.add_argument("--output-tokens-col", default=None)
    p.add_argument("--model-col", default=None)
    p.add_argument("--train-ratio", type=float, default=0.60)
    p.add_argument("--validation-ratio", type=float, default=0.20)
    p.add_argument("--test-ratio", type=float, default=0.20)
    p.add_argument("--guard-seconds", type=float, default=30.0)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--drop-zero-output", action="store_true")
    return p


def norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def resolve_key(row: dict[str, str], explicit: str | None, candidates: tuple[str, ...], required: bool = True) -> str | None:
    if explicit:
        return explicit
    lookup = {norm(k): k for k in row}
    for candidate in candidates:
        key = norm(candidate)
        if key in lookup:
            return lookup[key]
    if required:
        raise ValueError(f"Cannot resolve column from candidates: {candidates}")
    return None


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_summary(name: str, rows: list[dict[str, str]], timestamp_col: str) -> dict[str, Any]:
    timestamps = [float(row[timestamp_col]) for row in rows]
    return {
        "name": name,
        "rows": len(rows),
        "start_timestamp": min(timestamps) if timestamps else None,
        "end_timestamp": max(timestamps) if timestamps else None,
        "duration": (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0,
    }


def main() -> None:
    args = parser().parse_args()
    total_ratio = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/validation/test ratios must sum to 1.0")

    with args.input.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError("Input trace is empty")

    timestamp_col = resolve_key(rows[0], args.timestamp_col, ("timestamp", "time", "arrival", "created_at"))
    input_col = resolve_key(rows[0], args.input_tokens_col, ("request tokens", "request_tokens", "input_tokens", "prompt_tokens"), required=False)
    output_col = resolve_key(rows[0], args.output_tokens_col, ("response tokens", "response_tokens", "output_tokens", "completion_tokens"), required=False)
    model_col = resolve_key(rows[0], args.model_col, ("model", "model_name", "engine", "type"), required=False)

    clean: list[dict[str, str]] = []
    for row in rows:
        timestamp = as_float(row.get(timestamp_col))
        if timestamp is None:
            continue
        if args.drop_zero_output and output_col:
            output_tokens = as_float(row.get(output_col))
            if output_tokens is None or output_tokens <= 0.0:
                continue
        clean.append(row)
        if args.max_rows is not None and len(clean) >= args.max_rows:
            break
    clean.sort(key=lambda r: float(r[timestamp_col]))
    if len(clean) < 3:
        raise ValueError("Need at least 3 valid rows after cleaning")

    min_t = float(clean[0][timestamp_col])
    max_t = float(clean[-1][timestamp_col])
    span = max(max_t - min_t, 1e-9)
    train_cut = min_t + span * args.train_ratio
    val_cut = min_t + span * (args.train_ratio + args.validation_ratio)
    guard = max(0.0, args.guard_seconds)

    train_rows = [r for r in clean if float(r[timestamp_col]) < train_cut - guard]
    val_rows = [r for r in clean if train_cut + guard <= float(r[timestamp_col]) < val_cut - guard]
    test_rows = [r for r in clean if float(r[timestamp_col]) >= val_cut + guard]

    outputs = {
        "train": args.output_dir / "trace_train.csv",
        "validation": args.output_dir / "trace_validation.csv",
        "test": args.output_dir / "trace_test.csv",
    }
    write_csv(outputs["train"], train_rows, fields)
    write_csv(outputs["validation"], val_rows, fields)
    write_csv(outputs["test"], test_rows, fields)

    manifest = {
        "input": str(args.input),
        "split_policy": "chronological",
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "guard_seconds": guard,
        "cut_timestamps": {
            "train_validation_cut": train_cut,
            "validation_test_cut": val_cut,
        },
        "columns": {
            "timestamp": timestamp_col,
            "input_tokens": input_col,
            "output_tokens": output_col,
            "model": model_col,
        },
        "splits": {
            "train": split_summary("train", train_rows, timestamp_col),
            "validation": split_summary("validation", val_rows, timestamp_col),
            "test": split_summary("test", test_rows, timestamp_col),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
