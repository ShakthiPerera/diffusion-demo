"""
Plots for norm/iso metrics vs training step using saved CSVs.

For each dataset/objective/run/reg, the script loads
`norm_iso_metrics_<dataset>_reg_<reg>.csv` and produces line plots of the
available metrics (norm_pred, norm_true, iso) plus the difference
(norm_pred - norm_true). Users can select datasets, objectives, runs, and
reg values, mirroring the other utilities in this repo.

Directory layout (created under --base_dir):
  results/norm_iso_plots/<dataset>/<objective>/
      run_1/
        norm_iso_vs_steps_norm_pred.png
        norm_iso_vs_steps_norm_true.png
        norm_iso_vs_steps_iso.png
        norm_iso_vs_steps_norm_diff.png
        summary_norm_iso.csv
      run_2/
      run_3/
      mean_run/
        norm_iso_vs_steps_*.png
        summary_norm_iso.csv
      smooth_run/
        norm_iso_vs_steps_*_smoothed.png

Usage examples
--------------
  python norm_iso_plots.py --base_dir outputs \
    --datasets central_banana --objectives iso --regs 0.0 0.3

  # Process all datasets/objectives
  python norm_iso_plots.py --base_dir outputs --datasets all --objectives all
"""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_METRICS = ["norm_pred", "norm_true", "iso"]
PLOT_METRICS = BASE_METRICS + ["norm_diff"]


# ------------------------------
# Discovery helpers
# ------------------------------

def discover_datasets(base_dir: Path) -> List[str]:
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir() and p.name != "results"])


def discover_objectives(dataset_dir: Path) -> List[str]:
    objs = set()
    for p in dataset_dir.iterdir():
        if p.is_dir() and "_run_" in p.name:
            objs.add(p.name.split("_run_")[0])
    return sorted(objs)


# ------------------------------
# IO
# ------------------------------

def ensure_outdir(base_dir: Path, dataset: str, objective: str, subfolder: str) -> Path:
    out = base_dir / "results" / "norm_iso_plots" / dataset / objective / subfolder
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_norm_iso_csv(reg_dir: Path, dataset: str, reg_str: str, min_step: Optional[int], max_step: Optional[int]) -> pd.DataFrame:
    csv_path = reg_dir / f"norm_iso_metrics_{dataset}_reg_{reg_str}.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=["step", *PLOT_METRICS])
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame(columns=["step", *PLOT_METRICS])
    needed = ["step", *BASE_METRICS]
    if not all(c in df.columns for c in needed) or df.empty:
        return pd.DataFrame(columns=["step", *PLOT_METRICS])
    df = df.sort_values("step").reset_index(drop=True)
    if (min_step is not None) or (max_step is not None):
        lo = min_step if min_step is not None else int(df["step"].min())
        hi = max_step if max_step is not None else int(df["step"].max())
        df = df[(df["step"] >= lo) & (df["step"] <= hi)].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["step", *PLOT_METRICS])
    # Compute norm difference
    df["norm_diff"] = df["norm_pred"].astype(float) - df["norm_true"].astype(float)
    cols = ["step", *PLOT_METRICS]
    return df[cols]


def collect_norm_iso_one_run(
    dataset_dir: Path,
    dataset_name: str,
    objective: str,
    run_id: int,
    regs_filter: Optional[Iterable[float]],
    min_step: Optional[int],
    max_step: Optional[int],
) -> Dict[float, pd.DataFrame]:
    reg_to_df: Dict[float, pd.DataFrame] = {}
    run_dir = dataset_dir / f"{objective}_run_{run_id}"
    if not run_dir.exists():
        return reg_to_df
    regs_set = set(float(r) for r in regs_filter) if regs_filter is not None else None
    for reg_dir in sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("reg_")]):
        try:
            reg_str = reg_dir.name.split("reg_")[1]
            reg_val = float(reg_str)
        except Exception:
            continue
        if regs_set is not None and reg_val not in regs_set:
            continue
        df = read_norm_iso_csv(reg_dir, dataset_name, reg_str, min_step=min_step, max_step=max_step)
        if not df.empty:
            reg_to_df[reg_val] = df
    return reg_to_df


def collect_norm_iso_across_runs(
    dataset_dir: Path,
    dataset_name: str,
    objective: str,
    runs: Iterable[int],
    regs_filter: Optional[Iterable[float]],
    min_step: Optional[int],
    max_step: Optional[int],
) -> Dict[int, Dict[float, pd.DataFrame]]:
    out: Dict[int, Dict[float, pd.DataFrame]] = {}
    for r in runs:
        reg_map = collect_norm_iso_one_run(dataset_dir, dataset_name, objective, r, regs_filter, min_step, max_step)
        if reg_map:
            out[r] = reg_map
    return out


# ------------------------------
# Aggregation
# ------------------------------

def aggregate_mean_across_runs(run_to_df: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    bucket: Dict[int, Dict[str, List[float]]] = {}
    for df in run_to_df.values():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            step = int(row["step"])
            metric_bucket = bucket.setdefault(step, {m: [] for m in PLOT_METRICS})
            for m in PLOT_METRICS:
                if m in row and not pd.isna(row[m]):
                    metric_bucket[m].append(float(row[m]))
    rows = []
    for step, metric_vals in bucket.items():
        rec = {"step": step}
        for m in PLOT_METRICS:
            vals = metric_vals.get(m, [])
            if vals:
                rec[m] = float(np.mean(vals))
        rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=["step", *PLOT_METRICS])
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def reg_suffix(regs_filter: Optional[Iterable[float]]) -> str:
    if regs_filter is None:
        return ""
    regs = sorted(set(float(r) for r in regs_filter))
    return f"_regs-{'_'.join([str(r) for r in regs])}"


# ------------------------------
# Plotting helpers
# ------------------------------

def _set_axis_labels(ax: plt.Axes, metric: str, smoothed: bool = False) -> None:
    if metric == "norm_diff":
        ax.set_ylabel("norm_pred - norm_true")
        ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
        title_metric = "norm_diff"
    else:
        ax.set_ylabel(metric)
        title_metric = metric
    suffix = " (smoothed)" if smoothed else ""
    ax.set_title(f"{{}} — {title_metric.replace('_', ' ').capitalize()}{suffix}")


def plot_overlay(reg_to_df: Dict[float, pd.DataFrame], title_prefix: str, outdir: Path, filename_suffix: str = ""):
    if not reg_to_df:
        return
    regs = sorted(reg_to_df.keys())
    colors = plt.cm.tab10.colors
    for m in PLOT_METRICS:
        fig, ax = plt.subplots()
        for i, reg in enumerate(regs):
            df = reg_to_df[reg]
            if df is None or df.empty or (m not in df.columns):
                continue
            ax.plot(df["step"].values, df[m].values, marker='o', linewidth=1.4, label=f"reg={reg:g}", color=colors[i % len(colors)])
        if ax.lines:
            ax.set_xlabel("training step")
            if m == "norm_diff":
                ax.set_ylabel("norm_pred - norm_true")
                ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
                title_metric = "norm_diff"
            else:
                ax.set_ylabel(m)
                title_metric = m
            ax.set_title(f"{title_prefix} — {title_metric.replace('_', ' ').capitalize()}")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            fig.tight_layout()
            name = f"norm_iso_vs_steps_{m}{filename_suffix}.png"
            fig.savefig(outdir / name, dpi=200)
        plt.close(fig)


def _loess_fit(x: np.ndarray, y: np.ndarray, grid_x: np.ndarray, frac: float = 0.3) -> np.ndarray:
    x = x.astype(float)
    y = y.astype(float)
    n = len(x)
    if n == 0:
        return np.full_like(grid_x, np.nan, dtype=float)
    if n == 1:
        return np.full_like(grid_x, y[0], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    frac = float(np.clip(frac, 0.01, 1.0))
    k = max(2, int(np.ceil(frac * n)))
    ys = np.empty_like(grid_x, dtype=float)
    for i, gx in enumerate(grid_x):
        idx = np.searchsorted(x, gx)
        lo = max(0, idx - k // 2)
        hi = min(n, lo + k)
        lo = max(0, hi - k)
        xi = x[lo:hi]
        yi = y[lo:hi]
        d = np.abs(xi - gx)
        dmax = d.max() if d.size else 1.0
        if dmax == 0:
            ys[i] = yi.mean()
            continue
        u = d / dmax
        w = (1 - u**3)**3
        X = np.column_stack([np.ones_like(xi), xi - gx])
        W = np.diag(w)
        try:
            beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ yi)
            ys[i] = beta[0]
        except Exception:
            ys[i] = np.average(yi, weights=w)
    return ys


def plot_overlay_smoothed(
    run_to_reg_to_df: Dict[int, Dict[float, pd.DataFrame]],
    title_prefix: str,
    outdir: Path,
    filename_suffix: str = "",
    num_points: int = 400,
    frac: float = 0.3,
):
    if not run_to_reg_to_df:
        return
    reg_to_scatter: Dict[float, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for reg_map in run_to_reg_to_df.values():
        for reg, df in reg_map.items():
            for m in PLOT_METRICS:
                if df is None or df.empty or (m not in df.columns):
                    continue
                x = df["step"].values.astype(float)
                y = df[m].values.astype(float)
                d = reg_to_scatter.setdefault(reg, {})
                xs, ys = d.get(m, (np.array([], dtype=float), np.array([], dtype=float)))
                d[m] = (np.concatenate([xs, x]), np.concatenate([ys, y]))
    regs = sorted(reg_to_scatter.keys())
    if not regs:
        return
    colors = plt.cm.tab10.colors
    for m in PLOT_METRICS:
        fig, ax = plt.subplots()
        for i, reg in enumerate(regs):
            xs_all, ys_all = reg_to_scatter[reg].get(m, (None, None))
            if xs_all is None or xs_all.size == 0:
                continue
            gx = np.linspace(float(xs_all.min()), float(xs_all.max()), num_points)
            gy = _loess_fit(xs_all, ys_all, gx, frac=frac)
            ax.plot(gx, gy, linewidth=2.0, label=f"reg={reg:g}", color=colors[i % len(colors)])
        if ax.lines:
            ax.set_xlabel("training step")
            if m == "norm_diff":
                ax.set_ylabel("norm_pred - norm_true")
                ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
                title_metric = "norm_diff"
            else:
                ax.set_ylabel(m)
                title_metric = m
            ax.set_title(f"{title_prefix} — {title_metric.replace('_', ' ').capitalize()} (smoothed)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            fig.tight_layout()
            name = f"norm_iso_vs_steps_{m}_smoothed{filename_suffix}.png"
            fig.savefig(outdir / name, dpi=200)
        plt.close(fig)


# ------------------------------
# Orchestration
# ------------------------------

def process_objective(
    base_dir: Path,
    dataset: str,
    objective: str,
    runs: Iterable[int],
    regs_filter: Optional[Iterable[float]],
    min_step: Optional[int],
    max_step: Optional[int],
    smooth_frac: float,
    smooth_points: int,
):
    dataset_dir = base_dir / dataset
    run_to_reg_to_df = collect_norm_iso_across_runs(dataset_dir, dataset, objective, runs, regs_filter, min_step, max_step)

    # Mean across runs per reg
    reg_to_mean_df: Dict[float, pd.DataFrame] = {}
    regs = sorted({reg for reg_map in run_to_reg_to_df.values() for reg in reg_map.keys()})
    for reg in regs:
        run_to_df = {run_id: reg_map.get(reg) for run_id, reg_map in run_to_reg_to_df.items()}
        mean_df = aggregate_mean_across_runs(run_to_df)
        if not mean_df.empty:
            reg_to_mean_df[reg] = mean_df
    if reg_to_mean_df:
        outdir_mean = ensure_outdir(base_dir, dataset, objective, "mean_run")
        rows = []
        for reg, df in sorted(reg_to_mean_df.items()):
            tmp = df.copy()
            tmp.insert(0, "reg", reg)
            rows.append(tmp)
        df_long = pd.concat(rows, ignore_index=True)
        (outdir_mean / "summary_norm_iso.csv").write_text(df_long.to_csv(index=False))
        title = f"{dataset} · {objective} · mean_run"
        suffix = reg_suffix(regs_filter)
        if (min_step is not None) or (max_step is not None):
            suffix += f"_steps-{min_step if min_step is not None else 'min'}-{max_step if max_step is not None else 'max'}"
        plot_overlay(reg_to_mean_df, title_prefix=title, outdir=outdir_mean, filename_suffix=suffix)

    # Smoothed aggregate using all points from all runs
    if run_to_reg_to_df:
        outdir_smooth = ensure_outdir(base_dir, dataset, objective, "smooth_run")
        suffix = reg_suffix(regs_filter)
        if (min_step is not None) or (max_step is not None):
            suffix += f"_steps-{min_step if min_step is not None else 'min'}-{max_step if max_step is not None else 'max'}"
        plot_overlay_smoothed(
            run_to_reg_to_df,
            title_prefix=f"{dataset} · {objective} · smooth_run",
            outdir=outdir_smooth,
            filename_suffix=suffix,
            num_points=smooth_points,
            frac=smooth_frac,
        )


def main():
    ap = argparse.ArgumentParser(description="Plot norm/iso metrics vs training step from saved CSVs.")
    ap.add_argument("--base_dir", type=str, default="outputs", help="Root folder containing dataset folders.")
    ap.add_argument("--datasets", type=str, nargs="+", required=True, help="Datasets to process (names or 'all').")
    ap.add_argument("--objectives", type=str, nargs="*", default=None, help="Objectives to include (names or 'all'). If omitted, discover all present.")
    ap.add_argument("--runs", type=int, nargs="*", default=[1, 2, 3], help="Run IDs to include (default: 1 2 3).")
    ap.add_argument("--regs", type=float, nargs="*", default=None, help="Optional list of reg values to include (e.g., 0.0 0.3).")
    ap.add_argument("--min_step", type=int, default=None, help="Minimum step to include (optional).")
    ap.add_argument("--max_step", type=int, default=None, help="Maximum step to include (optional).")
    ap.add_argument("--smooth_frac", type=float, default=0.3, help="Fraction of points for LOESS smoothing (default: 0.3).")
    ap.add_argument("--smooth_points", type=int, default=400, help="Number of evaluation points for smoothing (default: 400).")
    args = ap.parse_args()

    base = Path(args.base_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Base dir not found: {base}")

    # Dataset selection
    if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
        datasets = discover_datasets(base)
    else:
        datasets = list(args.datasets)

    for ds in datasets:
        dataset_dir = base / ds
        if not dataset_dir.exists():
            print(f"[WARN] Dataset not found: {dataset_dir}")
            continue
        # Objective selection
        if args.objectives:
            if len(args.objectives) == 1 and args.objectives[0].lower() == "all":
                objectives = discover_objectives(dataset_dir)
            else:
                objectives = list(args.objectives)
        else:
            objectives = discover_objectives(dataset_dir)
            if not objectives:
                print(f"[WARN] No objectives discovered in {dataset_dir}")
                continue
        for obj in objectives:
            process_objective(
                base,
                ds,
                obj,
                args.runs,
                args.regs,
                args.min_step,
                args.max_step,
                smooth_frac=max(0.01, args.smooth_frac if args.smooth_frac else 0.3),
                smooth_points=max(50, args.smooth_points if args.smooth_points else 400),
            )
            print(f"[OK] {ds}/{obj}: norm/iso plots generated (mean and smooth runs).")


if __name__ == "__main__":
    main()
