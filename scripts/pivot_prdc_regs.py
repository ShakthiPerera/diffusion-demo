#!/usr/bin/env python3
"""Pivot PRDC metrics so DDPM (reg 0.0) and I-Diff (reg 0.3) sit on the same row."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

DDPM_REG = "0.0"
IDIFF_REG = "0.3"

METRICS = ("precision", "recall", "density", "coverage")

OUTPUT_HEADERS = [
    "Dataset",
    "Method",
    "Precision DDPM",
    "Precision I-Diff",
    "Recall DDPM",
    "Recall I-Diff",
    "Density DDPM",
    "Density I-Diff",
    "Coverage DDPM",
    "Coverage I-Diff",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine DDPM (reg 0.0) and I-Diff (reg 0.3) PRDC metrics into a single CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/results/prdc_final_25000_mean.csv"),
        help="Input CSV produced by average_prdc_runs.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/results/prdc_final_25000_ddpm_vs_idiff.csv"),
        help="Destination CSV with DDPM and I-Diff metrics side-by-side.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, str]]] = {}

    with args.input_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            dataset = row["dataset"]
            method = row["method"]
            reg = row.get("reg")
            if reg not in {DDPM_REG, IDIFF_REG}:
                continue
            grouped.setdefault((dataset, method), {})[reg] = row

    if not grouped:
        raise SystemExit("No matching records (reg 0.0 or 0.3) found in the input CSV.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(OUTPUT_HEADERS)
        for (dataset, method) in sorted(grouped.keys()):
            reg_map = grouped[(dataset, method)]
            ddpm_row = reg_map.get(DDPM_REG, {})
            idiff_row = reg_map.get(IDIFF_REG, {})

            def get_value(row: Dict[str, str], metric: str) -> str:
                return row.get(metric, "")

            writer.writerow(
                [
                    dataset,
                    method,
                    get_value(ddpm_row, "precision"),
                    get_value(idiff_row, "precision"),
                    get_value(ddpm_row, "recall"),
                    get_value(idiff_row, "recall"),
                    get_value(ddpm_row, "density"),
                    get_value(idiff_row, "density"),
                    get_value(ddpm_row, "coverage"),
                    get_value(idiff_row, "coverage"),
                ]
            )


if __name__ == "__main__":
    main()

