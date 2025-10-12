"""
Cross-objective PRDC percentage change analysis and plotting.

This script generalises the ISO-focused PRDC percentage change workflow to
compare multiple regularisation objectives (e.g. iso, mean_l2, var_l2, kl)
for a dataset. It supports two complementary visualisations:

1. reg_slice:
   For a user-specified regularisation value (e.g. reg=0.3), compute the PRDC
   % change versus the baseline reg=0.0 for every objective, export a CSV
   summarising the values (including the actual reg used per objective), and
   render bar charts of precision/recall/density/coverage across objectives.

2. best_per_objective:
   For each objective, locate the regularisation value that maximises the
   density % increase versus the baseline, export the chosen reg and % changes
   to a CSV, and render bar charts using that single row for all PRDC metrics.

PRDC data is gathered from checkpoint CSV files that are written during
training (see train.py). By default the script mirrors the existing ISO
workflow: ISO reg>0 entries use the checkpoint row that maximises a chosen
column (density by default). All other objectives fall back to the last
checkpoint row. You may customise this behaviour via CLI flags.

Directory layout (created under --base_dir):

  results/prdc_pct_change_cross/<dataset>/
      reg_slice/reg_<value>/
          pct_change.csv             (% changes + actual reg used per objective)
          pct_change_<metric>.png
      best_per_objective/
          pct_change_best.csv        (density-selected reg + % changes per objective)
          pct_change_best_<metric>.png

Usage example
-------------
  python prdc_pct_change_cross_objectives.py \
      --base_dir outputs \
      --datasets central_banana \
      --mode both --reg 0.3 \
      --objectives iso mean_l2 var_l2 kl \
      --runs 1 2 3
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PRDC_METRICS = ["precision", "recall", "density", "coverage"]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_checkpoint_prdc_best_by(
    csv_path: Path,
    select_by: str = "density",
    min_step: Optional[int] = None,
    max_step: Optional[int] = None,
) -> Dict[str, float]:
    """Return PRDC metrics at the checkpoint row with max(select_by)."""
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    needed = ["step", "precision", "recall", "density", "coverage"]
    if not all(c in df.columns for c in needed) or df.empty:
        return {}
    if (min_step is not None) or (max_step is not None):
        lo = min_step if min_step is not None else int(df["step"].min())
        hi = max_step if max_step is not None else int(df["step"].max())
        df_f = df[(df["step"] >= lo) & (df["step"] <= hi)]
        if df_f.empty:
            df_f = df
    else:
        df_f = df
    if select_by not in df_f.columns:
        return {}
    try:
        idx = df_f[select_by].idxmax()
    except Exception:
        return {}
    best = df_f.loc[idx]
    return {m: float(best[m]) for m in PRDC_METRICS if m in best}


def read_checkpoint_prdc_last(csv_path: Path) -> Dict[str, float]:
    """Return PRDC metrics recorded at the final checkpoint row."""
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
    return {m: float(last[m]) for m in PRDC_METRICS if m in last}


def discover_objectives(dataset_dir: Path) -> List[str]:
    """Infer available objectives from <objective>_run_<id> directories."""
    objs = set()
    for p in dataset_dir.iterdir():
        if p.is_dir() and "_run_" in p.name:
            objs.add(p.name.split("_run_")[0])
    return sorted(objs)


# ---------------------------------------------------------------------------
# Collection + aggregation
# ---------------------------------------------------------------------------

def collect_prdc_one_run(
    dataset_dir: Path,
    dataset_name: str,
    objective: str,
    run_id: int,
    strategy: str,
    select_by: str,
    min_step: Optional[int],
    max_step: Optional[int],
) -> Dict[float, Dict[str, float]]:
    """Gather {reg -> PRDC metrics} for a single run under the chosen strategy."""
    reg_to_metrics: Dict[float, Dict[str, float]] = {}
    run_dir = dataset_dir / f"{objective}_run_{run_id}"
    if not run_dir.exists():
        return reg_to_metrics

    reg_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("reg_")])
    for reg_dir in reg_dirs:
        try:
            reg_str = reg_dir.name.split("reg_")[1]
            reg_val = float(reg_str)
        except Exception:
            continue
        csv_path = reg_dir / "checkpoints" / f"checkpoint_prdc_metrics_{dataset_name}_reg_{reg_str}.csv"
        if abs(reg_val) < 1e-12:
            metrics = read_checkpoint_prdc_last(csv_path)
        else:
            if strategy == "best":
                metrics = read_checkpoint_prdc_best_by(
                    csv_path, select_by=select_by, min_step=min_step, max_step=max_step
                )
            else:
                metrics = read_checkpoint_prdc_last(csv_path)
        if metrics:
            reg_to_metrics[reg_val] = metrics
    return reg_to_metrics


def collect_prdc_across_runs(
    dataset_dir: Path,
    dataset_name: str,
    objective: str,
    runs: Iterable[int],
    strategy: str,
    select_by: str,
    min_step: Optional[int],
    max_step: Optional[int],
) -> Dict[int, Dict[float, Dict[str, float]]]:
    """Return {run_id -> {reg -> metrics}} for the given objective."""
    out: Dict[int, Dict[float, Dict[str, float]]] = {}
    for run_id in runs:
        run_map = collect_prdc_one_run(
            dataset_dir,
            dataset_name,
            objective,
            run_id,
            strategy=strategy,
            select_by=select_by,
            min_step=min_step,
            max_step=max_step,
        )
        if run_map:
            out[run_id] = run_map
    return out


def df_from_regmaps_mean(run_to_regmap: Dict[int, Dict[float, Dict[str, float]]]) -> pd.DataFrame:
    """Average metrics across runs for matching reg values."""
    bucket: Dict[float, Dict[str, List[float]]] = {}
    for regmap in run_to_regmap.values():
        for reg, metrics in regmap.items():
            dst = bucket.setdefault(reg, {})
            for metric, value in metrics.items():
                dst.setdefault(metric, []).append(value)
    rows = []
    for reg, metric_dict in bucket.items():
        row = {"reg": reg}
        for metric in PRDC_METRICS:
            values = metric_dict.get(metric, [])
            if values:
                row[metric] = float(np.mean(values))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=PRDC_METRICS)
    df = pd.DataFrame(rows).set_index("reg").sort_index()
    for metric in PRDC_METRICS:
        if metric not in df.columns:
            df[metric] = np.nan
    return df[PRDC_METRICS]


def pct_change_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Compute % change relative to baseline reg=0.0 (or smallest reg available)."""
    if df.empty:
        return df.copy()
    if 0.0 in df.index:
        baseline_reg = 0.0
    else:
        baseline_reg = float(df.index.min())
        print(f"[WARN] Baseline reg=0.0 not found; using reg={baseline_reg:g} as baseline")
    base = df.loc[baseline_reg].replace(0.0, np.nan)
    pct = (df.subtract(base) / base) * 100.0
    return pct


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def annotate_bars(ax, bars, fmt: str = "{:+.2f}%", extra_labels: Optional[List[str]] = None):
    """Annotate each bar with its height and optional extra label."""
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        if np.isnan(height):
            continue
        text = fmt.format(height)
        if extra_labels and idx < len(extra_labels) and extra_labels[idx]:
            text = f"{text}\n{extra_labels[idx]}"
        if height >= 0:
            offset = (0, 4)
            va = "bottom"
        else:
            offset = (0, -4)
            va = "top"
        ax.annotate(
            text,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )


def plot_metric_bars(
    data: pd.Series,
    title: str,
    ylabel: str,
    out_path: Path,
    extra_labels: Optional[List[str]] = None,
):
    """Render a simple bar plot for a Series indexed by objective."""
    fig, ax = plt.subplots(figsize=(8, 4))
    objectives = list(data.index)
    values = data.values
    bars = ax.bar(np.arange(len(objectives)), values, color="#4C72B0")
    ax.set_xticks(np.arange(len(objectives)))
    ax.set_xticklabels(objectives, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, color="black", linewidth=0.8)
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size:
        vmax = float(np.max(finite_vals))
        vmin = float(np.min(finite_vals))
        span = vmax - vmin
        pad = max(span * 0.1, 0.5)
        if span == 0.0:
            upper = vmax + 0.5
            lower = vmin - 0.5
        else:
            upper = vmax + pad
            lower = vmin - pad
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
        ax.set_ylim(lower, upper)
    annotate_bars(ax, bars, extra_labels=extra_labels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis modes
# ---------------------------------------------------------------------------

def select_row_for_reg(df: pd.DataFrame, target_reg: float, tol: float = 1e-9) -> Tuple[Optional[pd.Series], Optional[float]]:
    """Return the row for target_reg using a tolerance match."""
    if df.empty:
        return None, None
    for reg in df.index:
        if abs(float(reg) - target_reg) <= tol:
            return df.loc[reg], float(reg)
    return None, None


def format_reg(reg: float) -> str:
    """Consistent formatting for reg-dependent filenames."""
    if abs(reg) >= 1.0:
        return f"{reg:.1f}".rstrip("0").rstrip(".")
    return f"{reg:.3f}".rstrip("0").rstrip(".")


def run_reg_slice_mode(
    dataset: str,
    target_reg: float,
    objective_to_pct: Dict[str, pd.DataFrame],
    out_base: Path,
):
    """Plot PRDC % change across objectives for a chosen reg value."""
    collected_rows = {}
    actual_regs = {}
    for objective, df_pct in objective_to_pct.items():
        row, actual_reg = select_row_for_reg(df_pct, target_reg)
        if row is not None:
            collected_rows[objective] = row
            actual_regs[objective] = actual_reg
    if not collected_rows:
        print(f"[WARN] {dataset}: no objectives contained reg={target_reg:g}")
        return

    df_slice = pd.DataFrame(collected_rows).T[PRDC_METRICS]
    df_slice.index.name = "objective"
    slice_dir = out_base / dataset / "reg_slice" / f"reg_{format_reg(target_reg)}"
    slice_dir.mkdir(parents=True, exist_ok=True)
    reg_series = pd.Series({obj: actual_regs.get(obj, np.nan) for obj in df_slice.index}, name="reg")
    df_slice_out = df_slice.copy()
    df_slice_out.insert(0, "reg", reg_series)
    (slice_dir / "pct_change.csv").write_text(df_slice_out.round(3).to_csv())

    for metric in PRDC_METRICS:
        series = df_slice[metric].dropna()
        if series.empty:
            continue
        title = f"{dataset} · reg={format_reg(target_reg)} · {metric.capitalize()}"
        out_path = slice_dir / f"pct_change_{metric}.png"
        plot_metric_bars(
            data=series,
            title=title,
            ylabel="% change vs baseline",
            out_path=out_path,
            extra_labels=None,
        )
    print(f"[OK] {dataset}: reg_slice plots stored under {slice_dir}")


def run_best_per_objective_mode(
    dataset: str,
    objective_to_pct: Dict[str, pd.DataFrame],
    out_base: Path,
    drop_baseline: bool = True,
):
    """Plot the best % increase per metric for each objective."""
    best_values: Dict[str, Dict[str, float]] = {}
    best_regs: Dict[str, float] = {}
    for objective, df_pct in objective_to_pct.items():
        if df_pct.empty:
            continue
        df_candidates = df_pct.copy()
        if drop_baseline and 0.0 in df_candidates.index:
            df_candidates = df_candidates.drop(index=0.0, errors="ignore")
        if df_candidates.empty:
            continue
        density_col = df_candidates["density"].dropna()
        if density_col.empty:
            continue
        best_reg = float(density_col.idxmax())
        if best_reg not in df_candidates.index:
            continue
        best_row = df_candidates.loc[best_reg]
        best_values[objective] = {metric: float(best_row.get(metric, np.nan)) for metric in PRDC_METRICS}
        best_regs[objective] = best_reg

    if not best_values:
        print(f"[WARN] {dataset}: no best % increase data available.")
        return

    df_best = pd.DataFrame(best_values).T[PRDC_METRICS]
    df_best.index.name = "objective"
    reg_series = pd.Series(best_regs, name="reg")
    df_best_out = df_best.copy()
    df_best_out.insert(0, "reg", reg_series)

    best_dir = out_base / dataset / "best_per_objective"
    best_dir.mkdir(parents=True, exist_ok=True)
    (best_dir / "pct_change_best.csv").write_text(df_best_out.round(3).to_csv())

    for metric in PRDC_METRICS:
        series = df_best[metric].dropna()
        if series.empty:
            continue
        title = f"{dataset} · Best % increase · {metric.capitalize()}"
        out_path = best_dir / f"pct_change_best_{metric}.png"
        plot_metric_bars(
            data=series,
            title=title,
            ylabel="Max % change vs baseline",
            out_path=out_path,
            extra_labels=None,
        )
    print(f"[OK] {dataset}: best_per_objective plots stored under {best_dir}")


def clean_dataset_output_dir(dataset_out_dir: Path):
    """Remove any objective-specific folders so only reg_slice/best dirs remain."""
    if not dataset_out_dir.exists():
        return
    keep = {"reg_slice", "best_per_objective"}
    for child in dataset_out_dir.iterdir():
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare PRDC % change across objectives for selected datasets."
    )
    ap.add_argument("--base_dir", type=str, default="outputs", help="Root folder containing dataset directories.")
    ap.add_argument("--datasets", type=str, nargs="+", required=True, help="Datasets to process (e.g., central_banana).")
    ap.add_argument("--objectives", type=str, nargs="*", default=None, help="Objectives to include (omit to discover all).")
    ap.add_argument("--runs", type=int, nargs="*", default=[1, 2, 3], help="Run IDs to include when averaging.")
    ap.add_argument("--select_by", type=str, default="density", choices=PRDC_METRICS,
                    help="Column maximised when strategy=best (default: density).")
    ap.add_argument("--best_objectives", type=str, nargs="*", default=["iso"],
                    help="Objectives that should use the 'best' checkpoint selection strategy.")
    ap.add_argument("--min_step", type=int, default=None, help="Inclusive minimum checkpoint step to consider (optional).")
    ap.add_argument("--max_step", type=int, default=None, help="Inclusive maximum checkpoint step to consider (optional).")
    ap.add_argument("--mode", type=str, default="reg_slice", choices=["reg_slice", "best_per_objective", "both"],
                    help="Which visualisations to generate.")
    ap.add_argument("--reg", type=float, default=None,
                    help="Regularisation value for reg_slice mode (required if mode includes reg_slice).")
    return ap.parse_args()


def main():
    args = parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    if args.mode in ("reg_slice", "both") and args.reg is None:
        raise ValueError("reg_slice mode requires --reg.")

    best_objectives = set(args.best_objectives or [])
    runs = args.runs if args.runs else []
    if not runs:
        raise ValueError("At least one run ID must be provided.")

    out_base = base_dir / "results" / "prdc_pct_change_cross"

    for dataset in args.datasets:
        dataset_dir = base_dir / dataset
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

        objective_to_pct: Dict[str, pd.DataFrame] = {}
        for objective in objectives:
            strategy = "best" if objective in best_objectives else "last"
            run_to_regmap = collect_prdc_across_runs(
                dataset_dir=dataset_dir,
                dataset_name=dataset,
                objective=objective,
                runs=runs,
                strategy=strategy,
                select_by=args.select_by,
                min_step=args.min_step,
                max_step=args.max_step,
            )
            if not run_to_regmap:
                print(f"[WARN] {dataset}/{objective}: no checkpoint data found for runs {runs}")
                continue
            df_mean = df_from_regmaps_mean(run_to_regmap)
            if df_mean.empty:
                print(f"[WARN] {dataset}/{objective}: mean DataFrame empty, skipping")
                continue
            df_pct = pct_change_vs_baseline(df_mean)

            objective_to_pct[objective] = df_pct

        if not objective_to_pct:
            print(f"[WARN] {dataset}: no objectives produced PRDC summaries.")
            continue

        if args.mode in ("reg_slice", "both"):
            run_reg_slice_mode(
                dataset=dataset,
                target_reg=float(args.reg),
                objective_to_pct=objective_to_pct,
                out_base=out_base,
            )

        if args.mode in ("best_per_objective", "both"):
            run_best_per_objective_mode(
                dataset=dataset,
                objective_to_pct=objective_to_pct,
                out_base=out_base,
            )

        clean_dataset_output_dir(out_base / dataset)


if __name__ == "__main__":
    main()
