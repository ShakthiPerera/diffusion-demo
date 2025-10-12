"""
PRDC comparison of DDPM vs ISO regularisation across noise schedules.

For each dataset, the script expects the following directory layout under
``--base_dir`` (default: ``outputs``):

  <dataset>/iso_ddpm_<schedule>/reg_<reg_ddpm>/prdc_metrics_<dataset>_reg_<reg_ddpm>.csv
  <dataset>/iso_iso_<schedule>/reg_<reg_iso>/prdc_metrics_<dataset>_reg_<reg_iso>.csv

The CSV files must contain two columns, ``metric`` and ``value``. For each
schedule, the script pulls the DDPM baseline (reg=0.0 by default) and the ISO
run (reg=0.3 by default), computes ISO percentage change versus the DDPM values
of the same schedule, and renders single-bar charts per schedule. DDPM values
always use the final checkpoint row, whereas ISO values are taken from the
checkpoint row that maximises a chosen metric (default: density). Only
percentage-change bar charts are produced.

Outputs for each dataset are written to:
  results/prdc_vs_schedules/<dataset>/
    summary.csv
    prdc_pct_change_vs_schedule_<metric>.png

Usage example
-------------
  python prdc_ddpm_vs_iso_schedules.py \
      --base_dir outputs \
      --datasets central_banana \
      --schedules cosine linear quadratic sigmoid \
      --reg_iso 0.3 --reg_ddpm 0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PRDC_METRICS = ["precision", "recall", "density", "coverage"]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_prdc_metrics_csv(csv_path: Path) -> Dict[str, float]:
    """Load PRDC metric/value pairs from a CSV file."""
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    if {"metric", "value"} - set(df.columns):
        return {}
    return {row["metric"]: float(row["value"]) for _, row in df.iterrows() if row["metric"] in PRDC_METRICS}


def build_metric_path(
    base_dir: Path,
    dataset: str,
    objective_prefix: str,
    schedule: str,
    reg_value: float,
) -> Path:
    """Construct path to prdc_metrics CSV for the given configuration."""
    reg_str = str(reg_value)
    run_dir = base_dir / dataset / f"{objective_prefix}_{schedule}" / f"reg_{reg_str}"
    return run_dir / f"prdc_metrics_{dataset}_reg_{reg_str}.csv"


def checkpoint_metrics_path(run_dir: Path, dataset: str, reg_value: float) -> Path:
    reg_str = str(reg_value)
    return run_dir / "checkpoints" / f"checkpoint_prdc_metrics_{dataset}_reg_{reg_str}.csv"


def _clean_metric_floor(value: Optional[float]) -> Optional[float]:
    """Convert a floor value to float if possible; otherwise return None."""
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value_f):
        return None
    return value_f


def _metrics_from_row(row: pd.Series) -> Dict[str, float]:
    """Extract PRDC metrics from a dataframe row."""
    metrics: Dict[str, float] = {}
    for metric in PRDC_METRICS:
        if metric in row:
            try:
                metrics[metric] = float(row[metric])
            except (TypeError, ValueError):
                metrics[metric] = np.nan
    return metrics


def _has_precision_and_density(metrics: Dict[str, float]) -> bool:
    """Check that both precision and density values are usable for comparisons."""
    return (
        _clean_metric_floor(metrics.get("precision")) is not None
        and _clean_metric_floor(metrics.get("density")) is not None
    )


def read_checkpoint_table(
    csv_path: Path,
    min_step: Optional[int],
    max_step: Optional[int],
) -> Optional[pd.DataFrame]:
    """Load checkpoint PRDC metrics as a dataframe with optional step filtering."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    needed = {"step", *PRDC_METRICS}
    if not needed.issubset(df.columns) or df.empty:
        return None
    if (min_step is not None) or (max_step is not None):
        lo = min_step if min_step is not None else int(df["step"].min())
        hi = max_step if max_step is not None else int(df["step"].max())
        df_f = df[(df["step"] >= lo) & (df["step"] <= hi)]
        if df_f.empty:
            df_f = df
    else:
        df_f = df
    return df_f


def read_checkpoint_metrics(
    csv_path: Path,
    mode: str,
    select_by: str,
    min_step: Optional[int],
    max_step: Optional[int],
    metric_floors: Optional[Dict[str, float]],
) -> Dict[str, float]:
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    needed = {"step", *PRDC_METRICS}
    if not needed.issubset(df.columns) or df.empty:
        return {}
    if (min_step is not None) or (max_step is not None):
        lo = min_step if min_step is not None else int(df["step"].min())
        hi = max_step if max_step is not None else int(df["step"].max())
        df_f = df[(df["step"] >= lo) & (df["step"] <= hi)]
        if df_f.empty:
            df_f = df
    else:
        df_f = df
    if metric_floors:
        floors: Dict[str, float] = {
            metric: floor
            for metric, floor in (
                (m, _clean_metric_floor(v)) for m, v in metric_floors.items()
            )
            if floor is not None and metric in df_f.columns
        }
        if floors:
            mask = np.ones(len(df_f), dtype=bool)
            for metric, floor in floors.items():
                mask &= df_f[metric] > floor
            df_filtered = df_f[mask]
            if df_filtered.empty:
                return {}
            df_f = df_filtered
    if mode == "last":
        row = df_f.sort_values("step").iloc[-1]
    elif mode == "best":
        if select_by not in df_f.columns:
            return {}
        try:
            idx = df_f[select_by].idxmax()
        except Exception:
            return {}
        row = df_f.loc[idx]
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    return {m: float(row[m]) for m in PRDC_METRICS if m in row}


def load_metrics_with_fallback(
    run_dir: Path,
    dataset: str,
    reg_value: float,
    checkpoint_mode: str,
    select_by: str,
    min_step: Optional[int],
    max_step: Optional[int],
    metric_floors: Optional[Dict[str, float]],
) -> Dict[str, float]:
    ckpt_path = checkpoint_metrics_path(run_dir, dataset, reg_value)
    metrics = read_checkpoint_metrics(
        ckpt_path,
        mode=checkpoint_mode,
        select_by=select_by,
        min_step=min_step,
        max_step=max_step,
        metric_floors=metric_floors,
    )
    if metrics:
        pass
    else:
        csv_path = run_dir / f"prdc_metrics_{dataset}_reg_{reg_value}.csv"
        metrics = read_prdc_metrics_csv(csv_path)
    if metrics and metric_floors:
        for metric, floor in (
            (m, _clean_metric_floor(v)) for m, v in metric_floors.items()
        ):
            if floor is None:
                continue
            value = metrics.get(metric)
            if value is None:
                return {}
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return {}
            if np.isnan(value_f) or value_f <= floor:
                return {}
    return metrics


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def annotate_bars(ax, bars, fmt: str = "{:+.2f}%"):
    """Label each bar with its value, offsetting based on the sign."""
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        text = fmt.format(height)
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


def set_axis_ylim(ax, values: np.ndarray):
    """Expand y-axis limits to avoid bars touching the plot border."""
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return
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


def plot_pct_change(
    metric: str,
    schedules: List[str],
    iso_changes: np.ndarray,
    out_path: Path,
    dataset: str,
):
    """Plot ISO percent change vs schedule."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(schedules))
    bars = ax.bar(x, iso_changes, color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(schedules, rotation=20, ha="right")
    ax.set_ylabel("% change vs DDPM (same schedule)")
    ax.set_title(f"{dataset} · {metric.capitalize()} · ISO vs DDPM")
    set_axis_ylim(ax, iso_changes)
    ax.axhline(0.0, color="black", linewidth=0.8)
    annotate_bars(ax, bars)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_schedule_tables(
    base_dir: Path,
    dataset: str,
    schedules: Iterable[str],
    reg_ddpm: float,
    reg_iso: float,
    iso_select_by: str,
    min_step: Optional[int],
    max_step: Optional[int],
) -> pd.DataFrame:
    """Collect DDPM and ISO metrics for each schedule."""
    schedules = list(dict.fromkeys(schedules))  # preserve order, drop dups

    overrides = {
        ("swiss_roll", "linear"): {
            "ddpm": Path("iso_run_1") / f"reg_{reg_ddpm}",
            "iso": Path("iso_run_1") / f"reg_{reg_iso}",
        }
    }

    def schedule_dirs(schedule_name: str) -> tuple[Path, Path]:
        override = overrides.get((dataset, schedule_name))
        if override:
            ddpm_dir = base_dir / dataset / override["ddpm"]
            iso_dir = base_dir / dataset / override["iso"]
        else:
            ddpm_dir = base_dir / dataset / f"iso_ddpm_{schedule_name}" / f"reg_{reg_ddpm}"
            iso_dir = base_dir / dataset / f"iso_iso_{schedule_name}" / f"reg_{reg_iso}"
        return ddpm_dir, iso_dir

    rows: List[Dict[str, float]] = []
    for schedule in schedules:
        ddpm_dir, iso_dir = schedule_dirs(schedule)
        ddpm_metrics_base = load_metrics_with_fallback(
            ddpm_dir,
            dataset,
            reg_ddpm,
            checkpoint_mode="last",
            select_by=iso_select_by,
            min_step=min_step,
            max_step=max_step,
            metric_floors=None,
        )
        ddpm_ckpt_df = read_checkpoint_table(
            checkpoint_metrics_path(ddpm_dir, dataset, reg_ddpm),
            min_step=min_step,
            max_step=max_step,
        )

        ddpm_candidates: List[Dict[str, float]] = []
        if ddpm_metrics_base:
            ddpm_candidates.append(ddpm_metrics_base)
        if ddpm_ckpt_df is not None:
            df_sorted = ddpm_ckpt_df.sort_values("step", ascending=False)
            for _, row in df_sorted.iterrows():
                metric_row = _metrics_from_row(row)
                if metric_row:
                    ddpm_candidates.append(metric_row)

        selected_ddpm_metrics: Optional[Dict[str, float]] = None
        iso_metrics: Optional[Dict[str, float]] = None
        for ddpm_candidate in ddpm_candidates:
            if not _has_precision_and_density(ddpm_candidate):
                continue
            metric_floors = {
                "precision": ddpm_candidate.get("precision"),
                "density": ddpm_candidate.get("density"),
            }
            iso_candidate = load_metrics_with_fallback(
                iso_dir,
                dataset,
                reg_iso,
                checkpoint_mode="best",
                select_by=iso_select_by,
                min_step=min_step,
                max_step=max_step,
                metric_floors=metric_floors,
            )
            if iso_candidate:
                selected_ddpm_metrics = ddpm_candidate
                iso_metrics = iso_candidate
                break

        if selected_ddpm_metrics is None or iso_metrics is None:
            print(f"[WARN] {dataset}/{schedule}: missing PRDC metrics (ddpm_dir={ddpm_dir.exists()}, iso_dir={iso_dir.exists()})")
            continue
        ddpm_metrics = selected_ddpm_metrics
        row = {"schedule": schedule}
        for metric in PRDC_METRICS:
            ddpm_val = ddpm_metrics.get(metric, np.nan)
            iso_val = iso_metrics.get(metric, np.nan)
            row[f"{metric}_ddpm"] = ddpm_val
            row[f"{metric}_iso"] = iso_val
            iso_pct = np.nan
            if not np.isnan(ddpm_val) and ddpm_val != 0 and not np.isnan(iso_val):
                iso_pct = ((iso_val - ddpm_val) / ddpm_val) * 100.0
            row[f"{metric}_iso_pct_change"] = iso_pct
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("schedule")
    return df.sort_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare PRDC metrics for ISO reg vs DDPM baselines across schedules.")
    ap.add_argument("--base_dir", type=str, default="outputs", help="Root directory that holds dataset folders.")
    ap.add_argument("--datasets", type=str, nargs="+", required=True, help="Datasets to process.")
    ap.add_argument("--schedules", type=str, nargs="*", default=["cosine", "linear", "quadratic", "sigmoid"],
                    help="Noise schedule names to evaluate.")
    ap.add_argument("--reg_ddpm", type=float, default=0.0, help="Regularisation value for DDPM runs (default: 0.0).")
    ap.add_argument("--reg_iso", type=float, default=0.3, help="Regularisation value for ISO runs (default: 0.3).")
    ap.add_argument("--iso_select_by", type=str, default="density", choices=PRDC_METRICS,
                    help="Metric used to pick the best ISO checkpoint row (default: density).")
    ap.add_argument("--min_step", type=int, default=None, help="Inclusive minimum checkpoint step to consider.")
    ap.add_argument("--max_step", type=int, default=None, help="Inclusive maximum checkpoint step to consider.")
    return ap.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    for dataset in args.datasets:
        df = compute_schedule_tables(
            base_dir=base_dir,
            dataset=dataset,
            schedules=args.schedules,
            reg_ddpm=args.reg_ddpm,
            reg_iso=args.reg_iso,
            iso_select_by=args.iso_select_by,
            min_step=args.min_step,
            max_step=args.max_step,
        )
        if df.empty:
            print(f"[WARN] {dataset}: no schedules produced valid PRDC metrics.")
            continue

        out_dir = base_dir / "results" / "prdc_vs_schedules" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.csv").write_text(df.round(4).to_csv())

        schedules = list(df.index)
        for metric in PRDC_METRICS:
            iso_pct = df[f"{metric}_iso_pct_change"].values.astype(float)
            plot_pct_change(
                metric=metric,
                schedules=schedules,
                iso_changes=iso_pct,
                out_path=out_dir / f"prdc_pct_change_vs_schedule_{metric}.png",
                dataset=dataset,
            )
        print(f"[OK] {dataset}: plots written to {out_dir}")


if __name__ == "__main__":
    main()
