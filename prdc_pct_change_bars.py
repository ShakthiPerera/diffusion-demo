"""
Bar plots of PRDC % change vs. reg=0.0 for selected datasets/objectives.

This script scans outputs produced by train.py and creates bar charts of the
percentage change of PRDC metrics with respect to the baseline reg=0.0 for:
  - Each individual run (run_1, run_2, run_3)
  - The mean across available runs (mean_run)

Directory layout (created under --base_dir):
  results/prdc_pct_change/<dataset>/<objective>/
      run_1/
        prdc_pct_change_vs_reg_precision.png
        prdc_pct_change_vs_reg_recall.png
        prdc_pct_change_vs_reg_density.png
        prdc_pct_change_vs_reg_coverage.png
        summary_means.csv                 (raw PRDC values for the run)
        summary_pct_change.csv            (% change vs baseline for the run)
      run_2/
      run_3/
      mean_run/
        prdc_pct_change_vs_reg_*.png     (mean across runs)
        summary_means.csv                 (mean PRDC across runs)
        summary_pct_change.csv            (% change vs baseline for the mean)

Inputs per run/reg (produced by train.py checkpointing):
  <base_dir>/<dataset>/<objective>_run_<i>/reg_<reg>/checkpoints/
      checkpoint_prdc_metrics_<dataset>_reg_<reg>.csv

For each reg/run, the script selects the training step with the highest value
in a chosen column (default: density). You may also restrict the search to an
inclusive step range via command-line options. For the baseline reg=0.0,
the script always uses the last checkpoint row (latest step), regardless of
selection settings.

Usage examples
--------------
  # Specific datasets and objectives
  python prdc_pct_change_bars.py --base_dir outputs \
    --datasets central_banana moon_scatter \
    --objectives iso mean_l2

  # Process whatever objectives exist (omit --objectives)
  python prdc_pct_change_bars.py --base_dir outputs --datasets central_banana

Notes
-----
  - If reg=0.0 is missing, the script falls back to the smallest available
    reg value as the baseline and logs a warning.
  - If some runs are missing, the mean is computed across the runs that exist.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
from typing import Dict, List, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------
# IO helpers
# ------------------------------

def read_checkpoint_prdc_best_by(
    csv_path: Path,
    select_by: str = "density",
    min_step: Optional[int] = None,
    max_step: Optional[int] = None,
) -> Dict[str, float]:
    """Read checkpoint PRDC CSV and return metrics at the step with max(select_by).

    Expects columns: step, precision, recall, density, coverage. If multiple
    rows share the same max value, the first occurrence is used. If a step
    range is provided, the search is restricted to rows with
    min_step <= step <= max_step; if that yields no rows, the full CSV is
    used as a fallback.
    """
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    needed = ["step", "precision", "recall", "density", "coverage"]
    if not all(c in df.columns for c in needed) or df.empty:
        return {}
    # Optional step window filter (inclusive)
    if (min_step is not None) or (max_step is not None):
        lo = min_step if min_step is not None else int(df["step"].min())
        hi = max_step if max_step is not None else int(df["step"].max())
        df_f = df[(df["step"] >= lo) & (df["step"] <= hi)]
        if df_f.empty:
            df_f = df
    else:
        df_f = df
    # choose row with maximum of the selected column
    if select_by not in df_f.columns:
        return {}
    try:
        idx = df_f[select_by].idxmax()
    except Exception:
        return {}
    best = df_f.loc[idx]
    out = {m: float(best[m]) for m in METRICS if m in best}
    return out


def read_checkpoint_prdc_last(csv_path: Path) -> Dict[str, float]:
    """Read checkpoint PRDC CSV and return metrics at the last training step.

    Expects columns: step, precision, recall, density, coverage.
    """
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    needed = ["step", "precision", "recall", "density", "coverage"]
    if not all(c in df.columns for c in needed) or df.empty:
        return {}
    df = df.sort_values("step")
    last = df.iloc[-1]
    return {m: float(last[m]) for m in METRICS if m in last}


def discover_objectives(dataset_dir: Path) -> List[str]:
    objs = set()
    for p in dataset_dir.iterdir():
        if p.is_dir() and "_run_" in p.name:
            objs.add(p.name.split("_run_")[0])
    return sorted(objs)


# ------------------------------
# Collectors
# ------------------------------

def collect_prdc_one_run(
    dataset_dir: Path,
    dataset_name: str,
    objective: str,
    run_id: int,
    select_by: str,
    min_step: Optional[int],
    max_step: Optional[int],
) -> Dict[float, Dict[str, float]]:
    """
    Returns: { reg_value(float) -> { metric -> value } } for a single run.
    """
    reg_to_metrics_run: Dict[float, Dict[str, float]] = {}
    run_dir = dataset_dir / f"{objective}_run_{run_id}"
    if not run_dir.exists():
        return reg_to_metrics_run

    for reg_dir in sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("reg_")]):
        try:
            reg_str = reg_dir.name.split("reg_")[1]
            reg_val = float(reg_str)
        except Exception:
            continue
        csv_path = reg_dir / "checkpoints" / f"checkpoint_prdc_metrics_{dataset_name}_reg_{reg_str}.csv"
        # For baseline reg=0.0 always take the last checkpoint row
        if abs(reg_val - 0.0) < 1e-12:
            met = read_checkpoint_prdc_last(csv_path)
        else:
            met = read_checkpoint_prdc_best_by(csv_path, select_by=select_by, min_step=min_step, max_step=max_step)
        if met:
            reg_to_metrics_run[reg_val] = met
    return reg_to_metrics_run


def collect_prdc_across_runs(
    dataset_dir: Path,
    dataset_name: str,
    objective: str,
    runs: Iterable[int],
    select_by: str,
    min_step: Optional[int],
    max_step: Optional[int],
) -> Dict[int, Dict[float, Dict[str, float]]]:
    """
    Returns: { run_id -> { reg_value(float) -> { metric -> value } } }
    """
    out = {}
    for r in runs:
        out[r] = collect_prdc_one_run(dataset_dir, dataset_name, objective, r, select_by, min_step, max_step)
    return out


# ------------------------------
# Aggregation + % change
# ------------------------------

METRICS = ["precision", "recall", "density", "coverage"]


def df_from_regmap_single(reg_to_metrics_run: Dict[float, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for reg, met in reg_to_metrics_run.items():
        row = {"reg": reg}
        for k, v in met.items():
            row[k] = v
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=METRICS)
    df = pd.DataFrame(rows).set_index("reg").sort_index()
    for col in METRICS:
        if col not in df.columns:
            df[col] = np.nan
    return df[METRICS]


def df_from_regmaps_mean(run_to_regmap: Dict[int, Dict[float, Dict[str, float]]]) -> pd.DataFrame:
    # {reg -> {metric -> [values over runs]}}
    bucket: Dict[float, Dict[str, List[float]]] = {}
    for regmap in run_to_regmap.values():
        for reg, met in regmap.items():
            b = bucket.setdefault(reg, {})
            for k, v in met.items():
                b.setdefault(k, []).append(v)
    rows = []
    for reg, met_dict in bucket.items():
        row = {"reg": reg}
        for m in METRICS:
            vals = met_dict.get(m, [])
            if vals:
                row[m] = float(np.mean(vals))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=METRICS)
    df = pd.DataFrame(rows).set_index("reg").sort_index()
    for col in METRICS:
        if col not in df.columns:
            df[col] = np.nan
    return df[METRICS]


def pct_change_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    # Prefer baseline at exactly 0.0; else use the smallest reg value
    if 0.0 in df.index:
        base_idx = 0.0
    else:
        base_idx = float(df.index.min())
        print(f"[WARN] Baseline reg=0.0 not found; using reg={base_idx:g} as baseline")
    base = df.loc[base_idx].replace(0.0, np.nan)  # avoid divide-by-zero
    return (df.subtract(base) / base) * 100.0


# ------------------------------
# Plotting
# ------------------------------

def _annotate_bar_values(ax, bars):
    for b in bars:
        height = b.get_height()
        if np.isnan(height):
            continue
        ax.annotate(f"{height:+.2f}%",
                    xy=(b.get_x() + b.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)


def plot_bars_per_metric(df_pct: pd.DataFrame, title_prefix: str, outdir: Path, filename_suffix: str = ""):
    regs = df_pct.index.values.astype(float)
    xticks = [f"{r:g}" for r in regs]
    for m in METRICS:
        if m not in df_pct.columns:
            continue
        fig = plt.figure()
        ax = fig.add_subplot(111)
        vals = df_pct[m].values
        bars = ax.bar(np.arange(len(regs)), vals)
        _annotate_bar_values(ax, bars)
        ax.set_xticks(np.arange(len(regs)))
        ax.set_xticklabels(xticks)
        ax.set_xlabel("reg value")
        ax.set_ylabel("% change vs reg=0.0")
        ax.set_title(f"{title_prefix} — {m.capitalize()}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        name = f"prdc_pct_change_vs_reg_{m}{filename_suffix}.png"
        fig.savefig(outdir / name, dpi=200)
        plt.close(fig)


# ------------------------------
# Orchestration
# ------------------------------

def ensure_outdir(base_dir: Path, dataset: str, objective: str, subfolder: str) -> Path:
    out = base_dir / "results" / "prdc_pct_change" / dataset / objective / subfolder
    out.mkdir(parents=True, exist_ok=True)
    return out


def process_objective(base_dir: Path, dataset: str, objective: str, runs: Iterable[int], select_by: str, min_step: Optional[int], max_step: Optional[int]):
    dataset_dir = base_dir / dataset
    run_to_regmap = collect_prdc_across_runs(dataset_dir, dataset, objective, runs, select_by, min_step, max_step)

    # Per-run outputs
    for run_id, regmap in run_to_regmap.items():
        df_run = df_from_regmap_single(regmap)
        if df_run.empty:
            continue
        outdir_run = ensure_outdir(base_dir, dataset, objective, f"run_{run_id}")
        df_pct_run = pct_change_vs_baseline(df_run)
        # Summaries
        (outdir_run / "summary_means.csv").write_text(df_run.to_csv())
        (outdir_run / "summary_pct_change.csv").write_text(df_pct_run.round(3).to_csv())
        # Plots per metric
        # Concise title for readability; keep details in filename only
        title = f"{dataset} · {objective} · run {run_id}"
        suffix = f"_by-{select_by}"
        if (min_step is not None) or (max_step is not None):
            suffix += f"_steps-{min_step if min_step is not None else 'min'}-{max_step if max_step is not None else 'max'}"
        plot_bars_per_metric(df_pct_run, title_prefix=title, outdir=outdir_run, filename_suffix=suffix)

    # Mean across runs (only where present)
    df_mean = df_from_regmaps_mean(run_to_regmap)
    if not df_mean.empty:
        outdir_mean = ensure_outdir(base_dir, dataset, objective, "mean_run")
        df_pct_mean = pct_change_vs_baseline(df_mean)
        (outdir_mean / "summary_means.csv").write_text(df_mean.to_csv())
        (outdir_mean / "summary_pct_change.csv").write_text(df_pct_mean.round(3).to_csv())
        # Concise title for mean across runs
        title = f"{dataset} · {objective} · mean_run"
        suffix = f"_by-{select_by}"
        if (min_step is not None) or (max_step is not None):
            suffix += f"_steps-{min_step if min_step is not None else 'min'}-{max_step if max_step is not None else 'max'}"
        plot_bars_per_metric(df_pct_mean, title_prefix=title, outdir=outdir_mean, filename_suffix=suffix)


def main():
    ap = argparse.ArgumentParser(description="Bar plots of PRDC % change vs reg=0.0 for selected datasets/objectives.")
    ap.add_argument("--base_dir", type=str, default="outputs", help="Root folder that contains <dataset> directories.")
    ap.add_argument("--datasets", type=str, nargs="+", required=True, help="Datasets to process (e.g., central_banana moon_scatter).")
    ap.add_argument("--objectives", type=str, nargs="*", default=None, help="Objectives to include (e.g., iso mean_l2). If omitted, discover all present.")
    ap.add_argument("--runs", type=int, nargs="*", default=[1, 2, 3], help="Run IDs to include (default: 1 2 3).")
    ap.add_argument("--select_by", type=str, default="density", choices=["precision", "recall", "density", "coverage"],
                    help="Column to maximise when selecting the checkpoint row (default: density).")
    ap.add_argument("--min_step", type=int, default=None, help="Inclusive minimum step to search (optional).")
    ap.add_argument("--max_step", type=int, default=None, help="Inclusive maximum step to search (optional).")
    args = ap.parse_args()

    base = Path(args.base_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Base dir not found: {base}")

    for ds in args.datasets:
        dataset_dir = base / ds
        if not dataset_dir.exists():
            print(f"[WARN] Dataset not found: {dataset_dir}")
            continue
        if args.objectives:
            objectives = list(args.objectives)
        else:
            objectives = discover_objectives(dataset_dir)
            if not objectives:
                print(f"[WARN] No objectives discovered in {dataset_dir}")
                continue
        for obj in objectives:
            process_objective(base, ds, obj, args.runs, args.select_by, args.min_step, args.max_step)
            print(f"[OK] {ds}/{obj}: bar plots saved (per-run and mean).")


if __name__ == "__main__":
    main()
