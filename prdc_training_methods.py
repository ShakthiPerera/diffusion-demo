"""
PRDC comparison across diffusion training methods (standard vs ISO, SNR vs ISO).

For each dataset this script loads PRDC checkpoint CSVs for:
  - Baseline DDPM training (reg=0.0)
  - ISO-regularised counterpart (reg>0)
  - Optional SNR-aware DDPM/ISO variants

ISO metrics are selected from the checkpoint row that maximises a chosen column
(default: density), while DDPM metrics use the final checkpoint row. The script
computes the percentage change of ISO metrics relative to their DDPM
counterpart for each training method and generates bar plots per PRDC metric.

Directory expectations under --base_dir (default: outputs):
  <dataset>/iso_run_<id>/reg_<reg>/checkpoints/checkpoint_prdc_metrics_<dataset>_reg_<reg>.csv
  <dataset>/iso_snr_ddpm/reg_<reg>/checkpoints/checkpoint_prdc_metrics_<dataset>_reg_<reg>.csv
  <dataset>/iso_snr_iso_reg<reg>/reg_<reg>/checkpoints/checkpoint_prdc_metrics_<dataset>_reg_<reg>.csv

Outputs per dataset are written to:
  results/prdc_vs_training_methods/<dataset>/
    summary.csv
    prdc_pct_change_vs_method_<metric>.png

Usage example
-------------
  python prdc_training_methods.py \
      --base_dir outputs \
      --datasets central_banana \
      --runs 1 \
      --methods standard snr \
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

def read_checkpoint_metrics(
    csv_path: Path,
    mode: str,
    select_by: str,
    min_step: Optional[int],
    max_step: Optional[int],
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
) -> Dict[str, float]:
    reg_str = str(reg_value)
    ckpt_path = run_dir / "checkpoints" / f"checkpoint_prdc_metrics_{dataset}_reg_{reg_str}.csv"
    metrics = read_checkpoint_metrics(
        ckpt_path,
        mode=checkpoint_mode,
        select_by=select_by,
        min_step=min_step,
        max_step=max_step,
    )
    if metrics:
        return metrics
    csv_path = run_dir / f"prdc_metrics_{dataset}_reg_{reg_str}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return {}
        if {"metric", "value"}.issubset(df.columns):
            return {row["metric"]: float(row["value"]) for _, row in df.iterrows() if row["metric"] in PRDC_METRICS}
    return {}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        offset = (0, 4) if height >= 0 else (0, -4)
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            f"{height:+.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )


def set_axis_limits(ax, values: np.ndarray):
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


def plot_metric(
    metric: str,
    methods: List[str],
    pct_values: np.ndarray,
    out_path: Path,
    dataset: str,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(methods))
    bars = ax.bar(x, pct_values, color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("% change vs DDPM")
    ax.set_title(f"{dataset} · {metric.capitalize()} · ISO uplift by method")
    set_axis_limits(ax, pct_values)
    ax.axhline(0.0, color="black", linewidth=0.8)
    annotate_bars(ax, bars)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def mean_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {}
    data = {}
    for met in metric_list:
        for k, v in met.items():
            data.setdefault(k, []).append(v)
    return {k: float(np.mean(vals)) for k, vals in data.items() if vals}


def compute_method_summary(
    dataset_dir: Path,
    dataset: str,
    method: str,
    runs: Iterable[int],
    reg_ddpm: float,
    reg_iso: float,
    iso_select_by: str,
    min_step: Optional[int],
    max_step: Optional[int],
) -> Optional[Dict[str, Dict[str, float]]]:
    ddpm_metrics_all: List[Dict[str, float]] = []
    iso_metrics_all: List[Dict[str, float]] = []

    if method == "standard":
        for run in runs:
            ddpm_dir = dataset_dir / f"iso_run_{run}" / f"reg_{reg_ddpm}"
            iso_dir = dataset_dir / f"iso_run_{run}" / f"reg_{reg_iso}"
            if not ddpm_dir.exists() or not iso_dir.exists():
                continue
            ddpm_metrics = load_metrics_with_fallback(
                ddpm_dir, dataset, reg_ddpm, checkpoint_mode="last",
                select_by=iso_select_by, min_step=min_step, max_step=max_step
            )
            iso_metrics = load_metrics_with_fallback(
                iso_dir, dataset, reg_iso, checkpoint_mode="best",
                select_by=iso_select_by, min_step=min_step, max_step=max_step
            )
            if ddpm_metrics and iso_metrics:
                ddpm_metrics_all.append(ddpm_metrics)
                iso_metrics_all.append(iso_metrics)
    elif method == "snr":
        ddpm_dir = dataset_dir / "iso_snr_ddpm" / f"reg_{reg_ddpm}"
        iso_dir = dataset_dir / f"iso_snr_iso_reg{reg_iso}" / f"reg_{reg_iso}"
        if ddpm_dir.exists() and iso_dir.exists():
            ddpm_metrics = load_metrics_with_fallback(
                ddpm_dir, dataset, reg_ddpm, checkpoint_mode="last",
                select_by=iso_select_by, min_step=min_step, max_step=max_step
            )
            iso_metrics = load_metrics_with_fallback(
                iso_dir, dataset, reg_iso, checkpoint_mode="best",
                select_by=iso_select_by, min_step=min_step, max_step=max_step
            )
            if ddpm_metrics and iso_metrics:
                ddpm_metrics_all.append(ddpm_metrics)
                iso_metrics_all.append(iso_metrics)
    else:
        raise ValueError(f"Unknown method '{method}'")

    if not ddpm_metrics_all or not iso_metrics_all:
        return None

    ddpm_mean = mean_metrics(ddpm_metrics_all)
    iso_mean = mean_metrics(iso_metrics_all)
    if not ddpm_mean or not iso_mean:
        return None

    summary = {"ddpm": ddpm_mean, "iso": iso_mean}
    return summary


def compute_pct_change(ddpm: Dict[str, float], iso: Dict[str, float]) -> Dict[str, float]:
    pct = {}
    for metric in PRDC_METRICS:
        base = ddpm.get(metric, np.nan)
        val = iso.get(metric, np.nan)
        if base is None or val is None or np.isnan(base) or base == 0.0 or np.isnan(val):
            pct[metric] = np.nan
        else:
            pct[metric] = ((val - base) / base) * 100.0
    return pct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare PRDC metrics across diffusion training methods.")
    ap.add_argument("--base_dir", type=str, default="outputs", help="Root directory containing dataset folders.")
    ap.add_argument("--datasets", type=str, nargs="+", required=True, help="Datasets to process.")
    ap.add_argument("--runs", type=int, nargs="*", default=[1], help="Run IDs to average for standard training.")
    ap.add_argument("--methods", type=str, nargs="*", default=["standard", "snr"],
                    choices=["standard", "snr"], help="Training methods to include.")
    ap.add_argument("--reg_ddpm", type=float, default=0.0, help="Regularisation value for DDPM baselines.")
    ap.add_argument("--reg_iso", type=float, default=0.3, help="Regularisation value for ISO runs.")
    ap.add_argument("--iso_select_by", type=str, default="density", choices=PRDC_METRICS,
                    help="Metric used to select the best ISO checkpoint row.")
    ap.add_argument("--min_step", type=int, default=None, help="Inclusive minimum checkpoint step to consider.")
    ap.add_argument("--max_step", type=int, default=None, help="Inclusive maximum checkpoint step to consider.")
    return ap.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    for dataset in args.datasets:
        dataset_dir = base_dir / dataset
        if not dataset_dir.exists():
            print(f"[WARN] Dataset not found: {dataset_dir}")
            continue

        method_summaries = {}
        method_pct = {}
        for method in args.methods:
            summary = compute_method_summary(
                dataset_dir=dataset_dir,
                dataset=dataset,
                method=method,
                runs=args.runs,
                reg_ddpm=args.reg_ddpm,
                reg_iso=args.reg_iso,
                iso_select_by=args.iso_select_by,
                min_step=args.min_step,
                max_step=args.max_step,
            )
            if summary is None:
                print(f"[WARN] {dataset}: method '{method}' missing data.")
                continue
            pct_change = compute_pct_change(summary["ddpm"], summary["iso"])
            method_summaries[method] = summary
            method_pct[method] = pct_change

        if not method_summaries:
            print(f"[WARN] {dataset}: no methods available for comparison.")
            continue

        out_dir = base_dir / "results" / "prdc_vs_training_methods" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for method, summary in method_summaries.items():
            row = {"method": method}
            for metric in PRDC_METRICS:
                row[f"{metric}_ddpm"] = summary["ddpm"].get(metric, np.nan)
                row[f"{metric}_iso"] = summary["iso"].get(metric, np.nan)
                row[f"{metric}_iso_pct_change"] = method_pct[method].get(metric, np.nan)
            rows.append(row)
        df_summary = pd.DataFrame(rows).set_index("method")
        (out_dir / "summary.csv").write_text(df_summary.round(4).to_csv())

        methods = list(method_pct.keys())
        for metric in PRDC_METRICS:
            values = np.array([method_pct[m].get(metric, np.nan) for m in methods], dtype=float)
            plot_metric(
                metric=metric,
                methods=methods,
                pct_values=values,
                out_path=out_dir / f"prdc_pct_change_vs_method_{metric}.png",
                dataset=dataset,
            )
        print(f"[OK] {dataset}: training method comparison saved to {out_dir}")


if __name__ == "__main__":
    main()

