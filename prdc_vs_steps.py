"""
Line plots of PRDC metrics vs training steps for selected datasets/objectives/regs.

This script scans outputs produced by train.py and creates line charts of PRDC
metrics across training steps, overlaying multiple regularization values on the
same plot. It supports per-run outputs and the mean across available runs.

Directory layout (created under --base_dir):
  results/prdc_vs_steps/<dataset>/<objective>/
      run_1/
        prdc_vs_steps_precision.png
        prdc_vs_steps_recall.png
        prdc_vs_steps_density.png
        prdc_vs_steps_coverage.png
        summary_means.csv                 (raw PRDC timeseries for the run)
        summary_pct_change.csv            (% change vs baseline reg per step)
      run_2/
      run_3/
      mean_run/
        prdc_vs_steps_*.png              (mean across runs)
        summary_means.csv                 (mean PRDC timeseries across runs)
        summary_pct_change.csv            (% change vs baseline for the mean)

  # Additionally, an aggregated smooth run is placed under:
  results/prdc_vs_steps/<dataset>/<objective>/smooth_run/
      prdc_vs_steps_*_smoothed.png
      prdc_diff_vs_steps_*_smoothed.png

Inputs per run/reg (produced by train.py checkpointing):
  <base_dir>/<dataset>/<objective>_run_<i>/reg_<reg>/checkpoints/
      checkpoint_prdc_metrics_<dataset>_reg_<reg>.csv

Usage examples
--------------
  # Specific dataset, objective and regs
  python prdc_vs_steps.py --base_dir outputs \
    --datasets central_banana --objectives iso --regs 0.0 0.3

  # Process all datasets and all objectives present
  python prdc_vs_steps.py --base_dir outputs --datasets all --objectives all
"""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Dict, List, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = ["precision", "recall", "density", "coverage"]


# ------------------------------
# IO helpers
# ------------------------------

def read_checkpoint_prdc_csv(csv_path: Path, min_step: Optional[int], max_step: Optional[int]) -> pd.DataFrame:
    """Load a PRDC checkpoint CSV to DataFrame with columns: step + METRICS.

    Applies optional inclusive step window: min_step <= step <= max_step.
    Returns empty DataFrame if file missing or malformed.
    """
    if not csv_path.exists():
        return pd.DataFrame(columns=["step", *METRICS])
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame(columns=["step", *METRICS])
    needed = ["step", *METRICS]
    if not all(c in df.columns for c in needed) or df.empty:
        return pd.DataFrame(columns=["step", *METRICS])
    df = df.sort_values("step").reset_index(drop=True)
    if (min_step is not None) or (max_step is not None):
        lo = min_step if min_step is not None else int(df["step"].min())
        hi = max_step if max_step is not None else int(df["step"].max())
        df = df[(df["step"] >= lo) & (df["step"] <= hi)].reset_index(drop=True)
    return df[["step", *METRICS]]


def discover_datasets(base_dir: Path) -> List[str]:
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir() and p.name != "results"])


def discover_objectives(dataset_dir: Path) -> List[str]:
    objs = set()
    for p in dataset_dir.iterdir():
        if p.is_dir() and "_run_" in p.name:
            objs.add(p.name.split("_run_")[0])
    return sorted(objs)


# ------------------------------
# Collectors
# ------------------------------

def collect_timeseries_one_run(
    dataset_dir: Path,
    dataset_name: str,
    objective: str,
    run_id: int,
    regs_filter: Optional[Iterable[float]],
    min_step: Optional[int],
    max_step: Optional[int],
) -> Dict[float, pd.DataFrame]:
    """Return mapping: reg_value -> DataFrame(step + METRICS) for a single run."""
    out: Dict[float, pd.DataFrame] = {}
    run_dir = dataset_dir / f"{objective}_run_{run_id}"
    if not run_dir.exists():
        return out
    regs_set = set(float(r) for r in regs_filter) if regs_filter is not None else None
    for reg_dir in sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("reg_")]):
        try:
            reg_str = reg_dir.name.split("reg_")[1]
            reg_val = float(reg_str)
        except Exception:
            continue
        if regs_set is not None and reg_val not in regs_set:
            continue
        csv_path = reg_dir / "checkpoints" / f"checkpoint_prdc_metrics_{dataset_name}_reg_{reg_str}.csv"
        df = read_checkpoint_prdc_csv(csv_path, min_step=min_step, max_step=max_step)
        if not df.empty:
            out[reg_val] = df
    return out


def collect_timeseries_across_runs(
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
        out[r] = collect_timeseries_one_run(dataset_dir, dataset_name, objective, r, regs_filter, min_step, max_step)
    return out


# ------------------------------
# Aggregation + % change
# ------------------------------

def mean_timeseries_across_runs(run_to_reg_to_df: Dict[int, Dict[float, pd.DataFrame]]) -> Dict[float, pd.DataFrame]:
    """Compute mean timeseries across runs for each reg, aligning by step (outer join)."""
    # Gather per-reg list of dfs
    bucket: Dict[float, List[pd.DataFrame]] = {}
    for reg_map in run_to_reg_to_df.values():
        for reg, df in reg_map.items():
            bucket.setdefault(reg, []).append(df)

    mean_map: Dict[float, pd.DataFrame] = {}
    for reg, dfs in bucket.items():
        if not dfs:
            continue
        # Union of steps across runs
        union_steps = sorted(set().union(*[set(df["step"].tolist()) for df in dfs]))
        # For each metric, align to union_steps and average
        data = {"step": union_steps}
        for m in METRICS:
            cols = []
            for df in dfs:
                s = df.set_index("step")[m]
                s = s.reindex(union_steps)
                cols.append(s)
            arr = pd.concat(cols, axis=1)
            data[m] = arr.mean(axis=1, skipna=True).values
        mean_map[reg] = pd.DataFrame(data)
    return mean_map


def pct_change_vs_baseline_timeseries(reg_to_df: Dict[float, pd.DataFrame]) -> Dict[float, pd.DataFrame]:
    """Compute % change vs baseline reg per step for each metric (baseline step-matched)."""
    if not reg_to_df:
        return {}
    # Prefer exact 0.0 else smallest reg as baseline
    regs_sorted = sorted(reg_to_df.keys())
    base_reg = 0.0 if 0.0 in reg_to_df else regs_sorted[0]
    base_df = reg_to_df[base_reg].set_index("step")
    out: Dict[float, pd.DataFrame] = {}
    for reg, df in reg_to_df.items():
        if reg == base_reg:
            # 0% line for baseline
            pct_df = df.copy()
            for m in METRICS:
                pct_df[m] = 0.0
            out[reg] = pct_df
            continue
        df_i = df.set_index("step")
        # align on steps by intersection; avoid divide-by-zero
        common_steps = df_i.index.intersection(base_df.index)
        if common_steps.empty:
            continue
        num = df_i.loc[common_steps, METRICS]
        den = base_df.loc[common_steps, METRICS].replace(0.0, np.nan)
        pct = (num.subtract(den) / den) * 100.0
        pct.insert(0, "step", common_steps)
        out[reg] = pct.reset_index(drop=True)
    return out


# ------------------------------
# Plotting
# ------------------------------

def plot_lines_overlay(reg_to_df: Dict[float, pd.DataFrame], title_prefix: str, outdir: Path, filename_suffix: str = ""):
    if not reg_to_df:
        return
    regs = sorted(reg_to_df.keys())
    colors = plt.cm.tab10.colors
    for m in METRICS:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, reg in enumerate(regs):
            df = reg_to_df[reg]
            if m not in df.columns or df.empty:
                continue
            ax.plot(df["step"].values, df[m].values, label=f"reg={reg:g}", color=colors[i % len(colors)])
        ax.set_xlabel("training step")
        ax.set_ylabel(m)
        ax.set_title(f"{title_prefix} — {m.capitalize()}")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        name = f"prdc_vs_steps_{m}{filename_suffix}.png"
        fig.savefig(outdir / name, dpi=200)
        plt.close(fig)


def _loess_fit(x: np.ndarray, y: np.ndarray, grid_x: np.ndarray, frac: float = 0.3) -> np.ndarray:
    """Lightweight LOWESS/LOESS smoother (locally weighted linear regression).

    - x, y: raw points (1D), not necessarily unique in x.
    - grid_x: points where to evaluate the smoothed curve.
    - frac: fraction of data to use in local regression window (0 < frac <= 1).

    Uses tri-cube weights; handles boundaries naturally and avoids artifacts from
    simple moving averages.
    """
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
    k = max(2, int(np.ceil(frac * n)))
    ys = np.empty_like(grid_x, dtype=float)
    for i, gx in enumerate(grid_x):
        # Find k nearest neighbors in x
        idx = np.searchsorted(x, gx)
        lo = max(0, idx - k // 2)
        hi = min(n, lo + k)
        lo = max(0, hi - k)
        xi = x[lo:hi]
        yi = y[lo:hi]
        # Normalize distances and compute tri-cube weights
        d = np.abs(xi - gx)
        dmax = d.max() if d.size else 1.0
        if dmax == 0:
            ys[i] = yi.mean()
            continue
        u = d / dmax
        w = (1 - u**3)**3
        # Weighted linear regression: yi ~ a + b*(xi - gx)
        X = np.column_stack([np.ones_like(xi), xi - gx])
        W = np.diag(w)
        try:
            beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ yi)
            ys[i] = beta[0]
        except Exception:
            ys[i] = np.average(yi, weights=w)
    return ys


def plot_lines_overlay_smoothed_aggregated_scatter(run_to_reg_to_df: Dict[int, Dict[float, pd.DataFrame]], title_prefix: str, outdir: Path, filename_suffix: str = "", num_points: int = 400, frac: float = 0.3):
    if not run_to_reg_to_df:
        return
    # Aggregate scatter points across runs per reg
    reg_to_scatter: Dict[float, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for run_map in run_to_reg_to_df.values():
        for reg, df in run_map.items():
            for m in METRICS:
                x = df["step"].values.astype(float)
                y = df[m].values.astype(float)
                d = reg_to_scatter.setdefault(reg, {})
                xs, ys = d.get(m, (np.array([], dtype=float), np.array([], dtype=float)))
                d[m] = (np.concatenate([xs, x]), np.concatenate([ys, y]))

    regs = sorted(reg_to_scatter.keys())
    colors = plt.cm.tab10.colors
    for m in METRICS:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, reg in enumerate(regs):
            xs_all, ys_all = reg_to_scatter[reg].get(m, (None, None))
            if xs_all is None or xs_all.size == 0:
                continue
            # Grid over full combined step range
            gx = np.linspace(float(xs_all.min()), float(xs_all.max()), num_points)
            gy = _loess_fit(xs_all, ys_all, gx, frac=frac)
            ax.plot(gx, gy, label=f"reg={reg:g}", color=colors[i % len(colors)])
        ax.set_xlabel("training step")
        ax.set_ylabel(m)
        ax.set_title(f"{title_prefix} — {m.capitalize()} (smoothed aggregate)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        name = f"prdc_vs_steps_{m}_smoothed{filename_suffix}.png"
        fig.savefig(outdir / name, dpi=200)
        plt.close(fig)


def compute_diff_vs_baseline_timeseries(reg_to_df: Dict[float, pd.DataFrame]) -> Dict[float, pd.DataFrame]:
    """Compute absolute difference vs baseline reg per step for each metric.

    Returns mapping reg -> DataFrame(step + METRICS) where values are (reg - baseline).
    Baseline reg has zeros for all metrics.
    """
    if not reg_to_df:
        return {}
    regs_sorted = sorted(reg_to_df.keys())
    base_reg = 0.0 if 0.0 in reg_to_df else regs_sorted[0]
    base_df = reg_to_df[base_reg].set_index("step")
    out: Dict[float, pd.DataFrame] = {}
    for reg, df in reg_to_df.items():
        if reg == base_reg:
            z = df.copy()
            for m in METRICS:
                z[m] = 0.0
            out[reg] = z
            continue
        df_i = df.set_index("step")
        common_steps = df_i.index.intersection(base_df.index)
        if common_steps.empty:
            continue
        num = df_i.loc[common_steps, METRICS]
        den = base_df.loc[common_steps, METRICS]
        diff = (num - den)
        diff.insert(0, "step", common_steps)
        out[reg] = diff.reset_index(drop=True)
    return out


def plot_diff_overlay(reg_to_df: Dict[float, pd.DataFrame], title_prefix: str, outdir: Path, filename_suffix: str = ""):
    diff_map = compute_diff_vs_baseline_timeseries(reg_to_df)
    if not diff_map:
        return
    regs = [r for r in sorted(diff_map.keys()) if r != 0.0] + ([0.0] if 0.0 in diff_map else [])
    colors = plt.cm.tab10.colors
    for m in METRICS:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, reg in enumerate(regs):
            df = diff_map[reg]
            if m not in df.columns or df.empty:
                continue
            label = f"reg={reg:g} - base"
            if reg == 0.0:
                label = "baseline (0)"
            ax.plot(df["step"].values, df[m].values, label=label, color=colors[i % len(colors)])
        ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_xlabel("training step")
        ax.set_ylabel(f"Δ {m}")
        ax.set_title(f"{title_prefix} — Δ vs reg=0.0 ({m.capitalize()})")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        name = f"prdc_diff_vs_steps_{m}{filename_suffix}.png"
        fig.savefig(outdir / name, dpi=200)
        plt.close(fig)


def plot_diff_overlay_smoothed_aggregated_scatter(run_to_reg_to_df: Dict[int, Dict[float, pd.DataFrame]], title_prefix: str, outdir: Path, filename_suffix: str = "", num_points: int = 400, frac: float = 0.3):
    if not run_to_reg_to_df:
        return
    # Aggregate scatter points across runs per reg
    reg_to_scatter: Dict[float, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for run_map in run_to_reg_to_df.values():
        for reg, df in run_map.items():
            for m in METRICS:
                x = df["step"].values.astype(float)
                y = df[m].values.astype(float)
                d = reg_to_scatter.setdefault(reg, {})
                xs, ys = d.get(m, (np.array([], dtype=float), np.array([], dtype=float)))
                d[m] = (np.concatenate([xs, x]), np.concatenate([ys, y]))

    regs = sorted(reg_to_scatter.keys())
    if not regs:
        return
    # Determine baseline
    base_reg = 0.0 if 0.0 in reg_to_scatter else regs[0]
    colors = plt.cm.tab10.colors
    for m in METRICS:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Build grid over combined steps of baseline
        bx, by = reg_to_scatter[base_reg].get(m, (None, None))
        if bx is None or bx.size == 0:
            plt.close(fig)
            continue
        gx = np.linspace(float(bx.min()), float(bx.max()), num_points)
        by_s = _loess_fit(bx, by, gx, frac=frac)
        for i, reg in enumerate(regs):
            if reg == base_reg:
                continue
            xs_all, ys_all = reg_to_scatter[reg].get(m, (None, None))
            if xs_all is None or xs_all.size == 0:
                continue
            gy = _loess_fit(xs_all, ys_all, gx, frac=frac)
            ax.plot(gx, gy - by_s, label=f"reg={reg:g} - base", color=colors[i % len(colors)])
        ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_xlabel("training step")
        ax.set_ylabel(f"Δ {m}")
        ax.set_title(f"{title_prefix} — Δ vs reg={base_reg:g} ({m.capitalize()}, smoothed aggregate)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        name = f"prdc_diff_vs_steps_{m}_smoothed{filename_suffix}.png"
        fig.savefig(outdir / name, dpi=200)
        plt.close(fig)


# ------------------------------
# Orchestration
# ------------------------------

def ensure_outdir(base_dir: Path, dataset: str, objective: str, subfolder: str) -> Path:
    out = base_dir / "results" / "prdc_vs_steps" / dataset / objective / subfolder
    out.mkdir(parents=True, exist_ok=True)
    return out


def reg_suffix(regs_filter: Optional[Iterable[float]]) -> str:
    if regs_filter is None:
        return ""
    regs = sorted(set(float(r) for r in regs_filter))
    return f"_regs-{'_'.join([str(r) for r in regs])}"


def process_objective(base_dir: Path, dataset: str, objective: str, runs: Iterable[int], regs_filter: Optional[Iterable[float]], min_step: Optional[int], max_step: Optional[int], make_pct_change_csv: bool = True):
    dataset_dir = base_dir / dataset
    run_to_reg_to_df = collect_timeseries_across_runs(dataset_dir, dataset, objective, runs, regs_filter, min_step, max_step)

    # Mean across runs (only where present)
    mean_map = mean_timeseries_across_runs(run_to_reg_to_df)
    if mean_map:
        outdir_mean = ensure_outdir(base_dir, dataset, objective, "mean_run")
        # Save long-form CSV
        rows = []
        for reg, df in sorted(mean_map.items()):
            tmp = df.copy()
            tmp.insert(0, "reg", reg)
            rows.append(tmp)
        df_long = pd.concat(rows, ignore_index=True)
        (outdir_mean / "summary_means.csv").write_text(df_long.to_csv(index=False))
        if make_pct_change_csv:
            pct_map = pct_change_vs_baseline_timeseries(mean_map)
            rows = []
            for reg, df in sorted(pct_map.items()):
                tmp = df.copy()
                tmp.insert(0, "reg", reg)
                rows.append(tmp)
            if rows:
                df_pct = pd.concat(rows, ignore_index=True)
                (outdir_mean / "summary_pct_change.csv").write_text(df_pct.round(3).to_csv(index=False))
        # Plots
        title = f"{dataset} · {objective} · mean_run"
        suffix = reg_suffix(regs_filter)
        if (min_step is not None) or (max_step is not None):
            suffix += f"_steps-{min_step if min_step is not None else 'min'}-{max_step if max_step is not None else 'max'}"
        plot_lines_overlay(mean_map, title_prefix=title, outdir=outdir_mean, filename_suffix=suffix)
        plot_diff_overlay(mean_map, title_prefix=title, outdir=outdir_mean, filename_suffix=suffix)
        # Aggregated smoothed curves using all raw points from all runs
        outdir_smooth = ensure_outdir(base_dir, dataset, objective, "smooth_run")
        plot_lines_overlay_smoothed_aggregated_scatter(run_to_reg_to_df, title_prefix=f"{dataset} · {objective} · smooth_run", outdir=outdir_smooth, filename_suffix=suffix)
        plot_diff_overlay_smoothed_aggregated_scatter(run_to_reg_to_df, title_prefix=f"{dataset} · {objective} · smooth_run", outdir=outdir_smooth, filename_suffix=suffix)


def main():
    ap = argparse.ArgumentParser(description="Line plots of PRDC vs training steps for selected datasets/objectives/regs.")
    ap.add_argument("--base_dir", type=str, default="outputs", help="Root folder that contains <dataset> directories.")
    ap.add_argument("--datasets", type=str, nargs="+", required=True, help="Datasets to process (names or 'all').")
    ap.add_argument("--objectives", type=str, nargs="*", default=None, help="Objectives to include (names or 'all'). If omitted, discover all present.")
    ap.add_argument("--runs", type=int, nargs="*", default=[1, 2, 3], help="Run IDs to include (default: 1 2 3).")
    ap.add_argument("--regs", type=float, nargs="*", default=None, help="Optional list of reg values to include (e.g., 0.0 0.3).")
    ap.add_argument("--min_step", type=int, default=None, help="Inclusive minimum step to include (optional).")
    ap.add_argument("--max_step", type=int, default=None, help="Inclusive maximum step to include (optional).")
    args = ap.parse_args()

    base = Path(args.base_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Base dir not found: {base}")

    # Dataset selection (support 'all')
    if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
        datasets = discover_datasets(base)
    else:
        datasets = list(args.datasets)

    for ds in datasets:
        dataset_dir = base / ds
        if not dataset_dir.exists():
            print(f"[WARN] Dataset not found: {dataset_dir}")
            continue
        # Objective selection (support 'all' or discovery if omitted)
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
            process_objective(base, ds, obj, args.runs, args.regs, args.min_step, args.max_step)
            print(f"[OK] {ds}/{obj}: timeseries plots saved (mean and smooth runs).")


if __name__ == "__main__":
    main()
