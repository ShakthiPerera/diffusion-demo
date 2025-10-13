"""
Compute PRDC percentage gains versus a baseline run and generate bar plots.

The script loads PRDC metric CSVs for a set of ISO comparison runs and
compares each against a baseline directory. Results are exported as a summary
CSV with percentage gains alongside bar plots for each PRDC metric.

Examples
--------
# Auto-discover baseline/targets for the dataset.
python src/analysis/prdc_pct_gain_vs_baseline.py --base_dir outputs --dataset central_banana

# Process every dataset under the base directory.
python src/analysis/prdc_pct_gain_vs_baseline.py --base_dir outputs --dataset all

# Explicit baseline/targets/labels.
python src/analysis/prdc_pct_gain_vs_baseline.py \
    --base_dir outputs \
    --dataset central_banana \
    --baseline_run central_banana/iso_iso_compare_baseline \
    --targets \
        central_banana/iso_bures_iso_compare \
        central_banana/iso_frob_iso_compare \
        central_banana/iso_iso_compare \
        central_banana/iso_logeig_iso_compare \
        central_banana/iso_split_iso_compare
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PRDC_METRICS = ["precision", "recall", "density", "coverage"]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def discover_reg_dirs(run_dir: Path) -> List[Tuple[float, str, Path]]:
    """Return [(reg_value, reg_str, path)] for reg_* sub-directories."""
    regs: List[Tuple[float, str, Path]] = []
    if not run_dir.exists():
        return regs
    for sub in sorted(run_dir.iterdir()):
        if not (sub.is_dir() and sub.name.startswith("reg_")):
            continue
        reg_str = sub.name.split("reg_")[-1]
        try:
            reg_val = float(reg_str)
        except ValueError:
            continue
        regs.append((reg_val, reg_str, sub))
    return regs


def select_reg_dir(
    run_dir: Path,
    desired_reg: Optional[float] = None,
    tolerance: float = 1e-6,
) -> Tuple[float, str, Path]:
    """
    Choose a reg directory either by matching desired_reg or by auto-discovery.

    When desired_reg is None the function expects a single reg_* directory and
    will return it. Otherwise the closest match within `tolerance` is used.
    """
    regs = discover_reg_dirs(run_dir)
    if not regs:
        raise FileNotFoundError(f"No reg_* directories found in {run_dir}")
    if desired_reg is None:
        if len(regs) == 1:
            return regs[0]
        raise ValueError(
            f"Multiple reg directories in {run_dir}. Please specify --reg."
        )
    for reg_val, reg_str, path in regs:
        if abs(reg_val - desired_reg) <= tolerance:
            return reg_val, reg_str, path
    available = ", ".join(f"{val:g}" for val, _, _ in regs)
    raise ValueError(
        f"reg={desired_reg} not found in {run_dir}. Available: {available}"
    )


def read_prdc_file(csv_path: Path) -> Dict[str, float]:
    """Load PRDC metrics from either metric/value or checkpoint CSVs."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to read {csv_path}: {exc}") from exc

    if {"metric", "value"}.issubset(df.columns):
        metrics: Dict[str, float] = {}
        for _, row in df.iterrows():
            key = str(row["metric"])
            if key in PRDC_METRICS:
                metrics[key] = float(row["value"])
        if not metrics:
            raise ValueError(f"No PRDC metrics found in {csv_path}")
        return metrics

    if {"step", *PRDC_METRICS}.issubset(df.columns):
        df_sorted = df.sort_values("step")
        last = df_sorted.iloc[-1]
        return {m: float(last[m]) for m in PRDC_METRICS}

    raise ValueError(
        f"Unsupported PRDC CSV format in {csv_path}. Columns: {df.columns.tolist()}"
    )


def load_prdc_metrics(
    run_dir: Path,
    dataset: str,
    reg_hint: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Locate the reg_* directory and load PRDC metrics.

    Returns the resolved reg value and a mapping metric -> value.
    """
    reg_val, reg_str, reg_dir = select_reg_dir(run_dir, desired_reg=reg_hint)
    csv_candidates = [
        reg_dir / f"prdc_metrics_{dataset}_reg_{reg_str}.csv",
        reg_dir / "checkpoints" / f"checkpoint_prdc_metrics_{dataset}_reg_{reg_str}.csv",
    ]
    for candidate in csv_candidates:
        if candidate.exists():
            return reg_val, read_prdc_file(candidate)

    raise FileNotFoundError(
        f"No PRDC CSV found for reg={reg_str} under {run_dir}. "
        "Expected prdc_metrics_* or checkpoint CSV."
    )


# ---------------------------------------------------------------------------
# Computation + plotting
# ---------------------------------------------------------------------------

def compute_pct_change(
    baseline: Dict[str, float],
    targets: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Return dataframe with % change per metric for each target."""
    records: List[Dict[str, float]] = []
    for name, metrics in targets.items():
        row: Dict[str, float] = {"method": name}
        for metric in PRDC_METRICS:
            base_val = baseline.get(metric, np.nan)
            tgt_val = metrics.get(metric, np.nan)
            if np.isnan(base_val) or base_val == 0.0:
                pct = np.nan
            else:
                pct = ((tgt_val - base_val) / base_val) * 100.0
            row[metric] = pct
        records.append(row)
    df = pd.DataFrame.from_records(records)
    return df.set_index("method")


def annotate_bars(ax, bars):
    """Label bar heights with signed % values."""
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):  # pragma: no cover - guard
            continue
        offset = 4 if height >= 0 else -6
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            f"{height:+.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )


def plot_pct_change(metric: str, series: pd.Series, out_path: Path) -> None:
    """Render bar chart for a single PRDC metric."""
    fig, ax = plt.subplots(figsize=(7, 4))
    values = series.values.astype(float)
    indices = np.arange(len(series))
    bars = ax.bar(indices, values, color="#4c72b0")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(indices, series.index, rotation=20, ha="right")
    ax.set_ylabel("% change vs baseline")
    ax.set_title(f"{metric.title()} % change")

    ymin = np.nanmin(values)
    ymax = np.nanmax(values)
    span = ymax - ymin
    pad = max(span * 0.1, 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)

    annotate_bars(ax, bars)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PRDC % gains for ISO comparison runs vs baseline."
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing experiment outputs (default: outputs).",
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        required=True,
        nargs="+",
        help=(
            "Dataset name(s) used in PRDC CSV filenames (e.g. central_banana). "
            "Use 'all' to process every dataset under base_dir."
        ),
    )
    parser.add_argument(
        "--baseline_run",
        type=Path,
        default=None,
        help="Path (relative to base_dir) for the baseline run directory. "
        "Defaults to <dataset>/iso_iso_compare_baseline.",
    )
    parser.add_argument(
        "--baseline_reg",
        type=float,
        default=None,
        help="Regularisation value for the baseline (auto-detected if omitted).",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        nargs="+",
        default=None,
        help="Run directories (relative to base_dir) to compare against the baseline.",
    )
    parser.add_argument(
        "--target_regs",
        type=float,
        nargs="*",
        default=None,
        help="Optional list of regularisation values for each target run.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional display names for the target runs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Output directory (relative to base_dir). "
            "Defaults to results/prdc_pct_change_iso_metrics/<dataset>/"
        ),
    )
    return parser.parse_args()


def discover_baseline_and_targets(
    dataset_dir: Path, baseline_rel: Optional[Path]
) -> Tuple[Path, List[Path]]:
    """
    Determine baseline and comparison directories for a dataset.

    If baseline_rel is provided it is returned (after validation). Otherwise
    the function expects <dataset>/iso_iso_compare_baseline to exist.
    Target runs are auto-discovered as sub-directories that end with
    '_iso_compare' (excluding the baseline).
    """
    if baseline_rel is not None:
        baseline_dir = dataset_dir.parent / baseline_rel
    else:
        baseline_dir = dataset_dir / "iso_iso_compare_baseline"
    if not baseline_dir.exists():
        raise FileNotFoundError(
            f"Baseline directory not found: {baseline_dir}. "
            "Pass --baseline_run explicitly if it lives elsewhere."
        )

    target_dirs: List[Path] = []
    for sub in sorted(dataset_dir.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if name == "iso_iso_compare_baseline":
            continue
        if name.endswith("_iso_compare"):
            target_dirs.append(sub)

    if not target_dirs:
        raise FileNotFoundError(
            "Could not auto-discover any '*_iso_compare' runs under "
            f"{dataset_dir}. Provide --targets explicitly."
        )

    return baseline_dir, target_dirs


def process_dataset(
    dataset: str,
    base_dir: Path,
    args: argparse.Namespace,
) -> None:
    dataset_dir = base_dir / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if args.targets is None:
        baseline_dir, target_dirs = discover_baseline_and_targets(
            dataset_dir=dataset_dir, baseline_rel=args.baseline_run
        )
        # Convert to paths relative to base_dir for downstream consistency.
        baseline_rel = baseline_dir.relative_to(base_dir)
        target_rels = [t.relative_to(base_dir) for t in target_dirs]
        print(f"[{dataset}] Auto-selected baseline: {baseline_rel}")
        print(f"[{dataset}] Auto-selected targets:")
        for rel in target_rels:
            print(f"  - {rel}")
        baseline_dir = base_dir / baseline_rel
        target_dirs = [base_dir / rel for rel in target_rels]
        labels: Optional[List[str]] = None
        target_regs: List[Optional[float]] = [None] * len(target_dirs)
    else:
        baseline_dir = (
            base_dir / args.baseline_run
            if args.baseline_run is not None
            else base_dir / dataset / "iso_iso_compare_baseline"
        )
        target_dirs = [base_dir / t for t in args.targets]
        labels = args.labels
        if args.target_regs is not None:
            if len(args.target_regs) != len(target_dirs):
                raise ValueError("Number of target_regs values must match targets.")
            target_regs = list(args.target_regs)
        else:
            target_regs = [None] * len(target_dirs)

    if labels is not None and len(labels) != len(target_dirs):
        raise ValueError("Number of labels must match number of target directories.")

    if args.output_dir is not None:
        if args.multiple_datasets:
            output_dir = base_dir / args.output_dir / dataset
        else:
            output_dir = base_dir / args.output_dir
    else:
        output_dir = (
            base_dir
            / "results"
            / "prdc_pct_change_iso_metrics"
            / dataset
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline metrics
    baseline_reg, baseline_metrics = load_prdc_metrics(
        baseline_dir, dataset=dataset, reg_hint=args.baseline_reg
    )

    # Load target metrics
    target_metrics: Dict[str, Dict[str, float]] = {}
    target_info: List[Tuple[str, float, Dict[str, float]]] = []

    for idx, (run_dir, reg_hint) in enumerate(zip(target_dirs, target_regs)):
        name = (
            labels[idx]
            if labels is not None
            else run_dir.name
        )
        reg_value, metrics = load_prdc_metrics(run_dir, dataset=dataset, reg_hint=reg_hint)
        target_metrics[name] = metrics
        target_info.append((name, reg_value, metrics))

    pct_df = compute_pct_change(baseline_metrics, target_metrics)
    pct_df.index.name = "method"

    # Save summary CSV
    summary_csv = output_dir / "pct_gain_summary.csv"
    pct_df.to_csv(summary_csv)

    # Save absolute metrics CSV (baseline + targets)
    abs_records: List[Dict[str, float]] = [
        {"method": "baseline", "reg": baseline_reg, **baseline_metrics}
    ]
    for name, reg_value, metrics in target_info:
        rec = {"method": name, "reg": reg_value}
        rec.update(metrics)
        abs_records.append(rec)
    abs_df = pd.DataFrame.from_records(abs_records).set_index("method")
    abs_df.to_csv(output_dir / "absolute_metrics.csv")

    # Generate bar plots for each metric
    for metric in PRDC_METRICS:
        series = pct_df[metric]
        out_path = output_dir / f"pct_gain_{metric}.png"
        plot_pct_change(metric, series, out_path)

    print(f"[{dataset}] Wrote % gains to {summary_csv}")


def expand_dataset_names(base_dir: Path, requested: Iterable[str]) -> List[str]:
    """Resolve dataset tokens, expanding 'all' into discovered datasets."""
    result: List[str] = []
    for token in requested:
        if token.lower() == "all":
            discovered = sorted(
                p.name
                for p in base_dir.iterdir()
                if p.is_dir() and (p / "iso_iso_compare_baseline").exists()
            )
            result.extend(discovered)
        else:
            result.append(token)
    # Preserve order while removing duplicates.
    seen = set()
    ordered_unique: List[str] = []
    for name in result:
        if name not in seen:
            ordered_unique.append(name)
            seen.add(name)
    return ordered_unique


def main() -> None:
    args = parse_args()
    base_dir: Path = args.base_dir

    dataset_names = expand_dataset_names(base_dir, args.datasets)
    if not dataset_names:
        raise ValueError("No datasets resolved. Please check --dataset/--datasets value.")

    multiple = len(dataset_names) > 1
    setattr(args, "multiple_datasets", multiple)

    explicit_options = any(
        opt is not None
        for opt in (args.baseline_run, args.targets, args.labels, args.target_regs)
    )
    if multiple and explicit_options:
        raise ValueError(
            "Explicit baseline/target/label/reg arguments are only supported when "
            "processing a single dataset."
        )

    for dataset in dataset_names:
        process_dataset(dataset, base_dir=base_dir, args=args)


if __name__ == "__main__":
    main()
