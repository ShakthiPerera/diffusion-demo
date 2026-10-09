#!/usr/bin/env python3
"""Average PRDC metrics across runs for each dataset/method/reg combination."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

MetricSums = Dict[str, float]

ROUND_DIGITS = 4


def format_metric(value: float) -> str:
    return f"{value:.{ROUND_DIGITS}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average PRDC metrics across runs for each dataset/method/reg."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/results/prdc_final_25000.csv"),
        help="CSV containing per-run PRDC metrics.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/results/prdc_final_25000_mean.csv"),
        help="Destination CSV for the averaged metrics.",
    )
    return parser.parse_args()


def accumulate_metrics(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str, str, str], MetricSums]:
    sums: Dict[Tuple[str, str, str, str], MetricSums] = defaultdict(
        lambda: {
            "precision": 0.0,
            "recall": 0.0,
            "density": 0.0,
            "coverage": 0.0,
            "count": 0,
        }
    )
    for row in rows:
        dataset = row["dataset"]
        method = row["method"]
        reg = row.get("reg", "")
        step = row.get("step", "")
        key = (dataset, method, reg, step)
        entry = sums[key]
        try:
            entry["precision"] += float(row["precision"])
            entry["recall"] += float(row["recall"])
            entry["density"] += float(row["density"])
            entry["coverage"] += float(row["coverage"])
        except KeyError as exc:
            raise ValueError(f"Missing metric column in row: {row}") from exc
        entry["count"] += 1
    return sums


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    with args.input_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        sums = accumulate_metrics(reader)

    if not sums:
        raise SystemExit("No records found in the input CSV.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "method",
        "reg",
        "step",
        "run_count",
        "precision",
        "recall",
        "density",
        "coverage",
    ]

    sorted_keys = sorted(sums.keys())
    with args.output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, method, reg, step in sorted_keys:
            entry = sums[(dataset, method, reg, step)]
            count = entry["count"] or 1  # avoid division by zero
            writer.writerow(
                {
                    "dataset": dataset,
                    "method": method,
                    "reg": reg,
                    "step": step,
                    "run_count": count,
                    "precision": format_metric(entry["precision"] / count),
                    "recall": format_metric(entry["recall"] / count),
                    "density": format_metric(entry["density"] / count),
                    "coverage": format_metric(entry["coverage"] / count),
                }
            )


if __name__ == "__main__":
    main()
