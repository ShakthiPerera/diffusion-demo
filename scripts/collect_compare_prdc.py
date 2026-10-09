#!/usr/bin/env python3
"""Collect PRDC metrics from explicit method directories and pivot DDPM vs I-Diff."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

DDPM_REG = "0.0"
IDIFF_REG = "0.3"
TARGET_STEP = 25000
ROUND_DIGITS = 4

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
        description=(
            "Read PRDC metrics from explicitly listed method directories and "
            "produce a CSV with DDPM (reg 0.0) and I-Diff (reg 0.3) metrics side-by-side."
        )
    )
    parser.add_argument(
        "method_dirs",
        nargs="*",
        type=Path,
        help="Directories containing reg_*/checkpoints/checkpoint_prdc_metrics_*.csv files.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs"),
        help="Root directory to search when using --method-names.",
    )
    parser.add_argument(
        "--method-names",
        nargs="*",
        default=[],
        metavar="NAME",
        help="Directory names to search for under --root.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/results/prdc_compare_ddpm_vs_idiff.csv"),
        help="Destination CSV file.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=TARGET_STEP,
        help="Training step to extract from each checkpoint CSV.",
    )
    return parser.parse_args()


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


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.{ROUND_DIGITS}f}"


def normalize_method_name(path: Path) -> str:
    name = path.name
    if name.endswith("_baseline"):
        return name[: -len("_baseline")]
    return name


def find_metrics(method_dir: Path, dataset: str, reg: str, step: int) -> Optional[Dict[str, float]]:
    reg_dir = method_dir / f"reg_{reg}"
    if not reg_dir.is_dir():
        return None
    csv_path = reg_dir / "checkpoints" / f"checkpoint_prdc_metrics_{dataset}_reg_{reg}.csv"
    if not csv_path.is_file():
        return None
    return read_step(csv_path, step)


def collect_rows(
    method_dirs: Iterable[Path], step: int
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, Dict[str, float]]], Dict[str, Dict[str, float]]
]:
    aggregated: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    ddpm_baselines: Dict[str, Dict[str, float]] = {}
    for method_dir in method_dirs:
        canonical_dir = method_dir.resolve()
        if not canonical_dir.is_dir():
            continue
        dataset = canonical_dir.parent.name
        method = normalize_method_name(canonical_dir)
        is_baseline = method_dir.name.endswith("_baseline")
        ddpm_metrics = find_metrics(canonical_dir, dataset, DDPM_REG, step)
        idiff_metrics = find_metrics(canonical_dir, dataset, IDIFF_REG, step)
        if not ddpm_metrics and not idiff_metrics:
            continue
        entry = aggregated.setdefault((dataset, method), {"ddpm": {}, "idiff": {}})
        if ddpm_metrics:
            entry["ddpm"] = ddpm_metrics
            if is_baseline:
                ddpm_baselines.setdefault(dataset, ddpm_metrics)
        if idiff_metrics:
            entry["idiff"] = idiff_metrics
    return aggregated, ddpm_baselines


def expand_method_dirs(explicit_dirs: Iterable[Path], root: Path, method_names: Iterable[str]) -> List[Path]:
    collected: List[Path] = []
    seen: Set[Path] = set()

    for directory in explicit_dirs:
        resolved = directory.resolve()
        if not resolved.is_dir():
            continue
        if resolved not in seen:
            collected.append(resolved)
            seen.add(resolved)

    root_resolved = root.resolve()
    if method_names and not root_resolved.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    for name in method_names:
        matches = sorted(root_resolved.rglob(name))
        for match in matches:
            if match.is_dir() and match not in seen:
                collected.append(match)
                seen.add(match)

    return collected


def main() -> None:
    args = parse_args()
    method_dirs = expand_method_dirs(args.method_dirs, args.root, args.method_names)
    rows, ddpm_baselines = collect_rows(method_dirs, args.step)
    if not rows and not ddpm_baselines:
        raise SystemExit("No PRDC metrics found for the provided directories.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(OUTPUT_HEADERS)
        for (dataset, method), metrics in sorted(rows.items()):
            ddpm_metrics = metrics.get("ddpm", {})
            idiff_metrics = metrics.get("idiff", {})
            if not ddpm_metrics:
                fallback = ddpm_baselines.get(dataset, {})
                ddpm_metrics = fallback
            writer.writerow(
                [
                    dataset,
                    method,
                    format_metric(ddpm_metrics.get("precision")),
                    format_metric(idiff_metrics.get("precision")),
                    format_metric(ddpm_metrics.get("recall")),
                    format_metric(idiff_metrics.get("recall")),
                    format_metric(ddpm_metrics.get("density")),
                    format_metric(idiff_metrics.get("density")),
                    format_metric(ddpm_metrics.get("coverage")),
                    format_metric(idiff_metrics.get("coverage")),
                ]
            )


if __name__ == "__main__":
    main()
