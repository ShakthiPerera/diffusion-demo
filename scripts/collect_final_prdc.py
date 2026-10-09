#!/usr/bin/env python3

"""
Aggregate PRDC checkpoint metrics at a specific step across datasets and runs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the PRDC metrics at a given step for each dataset/method/run "
            "combination that contains runs 1, 2, and 3."
        )
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory that holds dataset outputs.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=25000,
        help="Training step to extract from each checkpoint CSV.",
    )
    parser.add_argument(
        "--regs",
        nargs="+",
        default=["0.0", "0.3"],
        metavar="REG",
        help="Regularization strengths to include (as strings matching folder names).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/results/prdc_final_25000.csv"),
        help="Where to write the aggregated CSV file.",
    )
    return parser.parse_args()


def collect_runs(dataset_dir: Path) -> Dict[str, Dict[int, Path]]:
    result: Dict[str, Dict[int, Path]] = {}
    for run_dir in dataset_dir.iterdir():
        if not run_dir.is_dir():
            continue
        name_parts = run_dir.name.rsplit("_run_", 1)
        if len(name_parts) != 2:
            continue
        base_name, run_suffix = name_parts
        try:
            run_idx = int(run_suffix)
        except ValueError:
            continue
        result.setdefault(base_name, {})[run_idx] = run_dir
    return result


def read_step(csv_path: Path, target_step: int) -> Optional[Dict[str, float]]:
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                step = int(float(row["step"]))
            except (KeyError, ValueError):
                continue
            if step != target_step:
                continue
            try:
                return {
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "density": float(row["density"]),
                    "coverage": float(row["coverage"]),
                }
            except (KeyError, ValueError):
                return None
    return None


def iter_dataset_rows(
    outputs_dir: Path, regs: Iterable[str], step: int
) -> Iterable[Dict[str, object]]:
    for dataset_dir in sorted(outputs_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "results":
            continue
        dataset = dataset_dir.name
        methods = collect_runs(dataset_dir)
        for method, runs in sorted(methods.items()):
            if not all(idx in runs for idx in (1, 2, 3)):
                continue
            for run_idx in (1, 2, 3):
                run_dir = runs[run_idx]
                for reg in regs:
                    reg_dir = run_dir / f"reg_{reg}"
                    if not reg_dir.is_dir():
                        continue
                    csv_path = (
                        reg_dir
                        / "checkpoints"
                        / f"checkpoint_prdc_metrics_{dataset}_reg_{reg}.csv"
                    )
                    if not csv_path.is_file():
                        continue
                    metrics = read_step(csv_path, step)
                    if metrics is None:
                        continue
                    yield {
                        "dataset": dataset,
                        "method": method,
                        "run": run_idx,
                        "reg": reg,
                        "step": step,
                        **metrics,
                    }


def main() -> None:
    args = parse_args()
    outputs_dir: Path = args.outputs_dir
    if not outputs_dir.is_dir():
        raise SystemExit(f"Outputs directory not found: {outputs_dir}")

    rows: List[Dict[str, object]] = list(
        iter_dataset_rows(outputs_dir, args.regs, args.step)
    )
    if not rows:
        raise SystemExit("No rows collected; check directory layout and arguments.")

    rows.sort(key=lambda r: (r["dataset"], r["method"], r["run"], r["reg"]))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "method",
        "run",
        "reg",
        "step",
        "precision",
        "recall",
        "density",
        "coverage",
    ]
    with args.output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

