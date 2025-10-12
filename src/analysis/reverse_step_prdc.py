"""
Compute PRDC vs reverse step from saved per-step samples and plot.

For each selected dataset/objective/run/reg, this script scans the
`samples/` folder under the reg directory for files named
`samples_step_<t>.npy` (generated points at reverse step `t`). For each
such step, it computes PRDC between the generated points and the original
real dataset found in the same reg directory (e.g., `dataset_*.npy`).
Note: the real dataset is NOT noised or diffused — comparisons are always
against the clean real dataset.

It caches the per-step PRDC results to a CSV under the reg directory so
subsequent runs are fast. Plots are saved in a separate results folder.

Directory layout (created under --base_dir):
  results/reverse_step_prdc/<dataset>/<objective>/
      run_1/
        reverse_step_prdc_precision.png
        reverse_step_prdc_recall.png
        reverse_step_prdc_density.png
        reverse_step_prdc_coverage.png
        summary_reverse_step_prdc.csv
      run_2/
      run_3/
      mean_run/
        reverse_step_prdc_*.png
        summary_reverse_step_prdc.csv

  Smoothed aggregate curves across runs are saved under:
      smooth_run/
        reverse_step_prdc_*_smoothed.png

Caching per reg:
  <base_dir>/<dataset>/<objective>_run_<i>/reg_<reg>/
    reverse_step_prdc_<dataset>_reg_<reg>.csv   (created if missing)

Usage examples
--------------
  python reverse_step_prdc.py --base_dir outputs \
    --datasets central_banana --objectives iso --runs 1 2 3 --regs 0.0 0.3

  # All datasets/objectives, cap to 2000 points for speed
  python reverse_step_prdc.py --base_dir outputs --datasets all --objectives all \
    --max_points 2000 --nearest_k 5
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from ..schedules.schedules import make_beta_schedule
from ..metrics.metrics import compute_prdc


METRICS = ["precision", "recall", "density", "coverage"]


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
# IO + cache
# ------------------------------

_STEP_RE = re.compile(r"samples_step_(\d+)\.npy$")


def list_sample_steps(samples_dir: Path, min_step: Optional[int], max_step: Optional[int]) -> List[int]:
    steps: List[int] = []
    for f in samples_dir.glob("samples_step_*.npy"):
        m = _STEP_RE.search(f.name)
        if not m:
            continue
        s = int(m.group(1))
        if (min_step is not None and s < min_step) or (max_step is not None and s > max_step):
            continue
        steps.append(s)
    return sorted(steps)


def read_settings(reg_dir: Path) -> Dict[str, str]:
    # Pick the first settings_*_reg_<reg>.txt file present
    files = sorted(reg_dir.glob("settings_*_reg_*.txt"))
    out: Dict[str, str] = {}
    if not files:
        return out
    txt = files[0].read_text().splitlines()
    for line in txt:
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def get_schedule_from_settings(settings: Dict[str, str]) -> Tuple[int, str]:
    num_steps = int(settings.get("num_diffusion_steps", 1000))
    schedule = settings.get("schedule", "linear").lower()
    return num_steps, schedule


def ensure_outdir(base_dir: Path, dataset: str, objective: str, subfolder: str) -> Path:
    out = base_dir / "results" / "reverse_step_prdc" / dataset / objective / subfolder
    out.mkdir(parents=True, exist_ok=True)
    return out


def cached_prdc_csv_path(reg_dir: Path, dataset: str, reg_str: str) -> Path:
    return reg_dir / f"reverse_step_prdc_{dataset}_reg_{reg_str}.csv"


# ------------------------------
# Core computation
# ------------------------------

def diffuse_real_at_step(x0: np.ndarray, step: int, betas: torch.Tensor, rng: np.random.Generator) -> np.ndarray:
    # Compute alpha_bar at step
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)[step].item()
    sqrt_ab = float(np.sqrt(alpha_bar))
    sqrt_one_minus_ab = float(np.sqrt(1.0 - alpha_bar))
    noise = rng.standard_normal(size=x0.shape).astype(np.float32)
    x_t = sqrt_ab * x0.astype(np.float32) + sqrt_one_minus_ab * noise
    return x_t


def compute_prdc_for_steps(reg_dir: Path, dataset: str, reg_str: str, nearest_k: int, max_points: Optional[int], min_step: Optional[int], max_step: Optional[int], seed: int = 123) -> pd.DataFrame:
    samples_dir = reg_dir / "samples"
    dataset_files = list(reg_dir.glob("dataset_*.npy"))
    if not samples_dir.exists() or not dataset_files:
        return pd.DataFrame(columns=["step", "reverse_step", *METRICS])
    settings = read_settings(reg_dir)
    num_steps, schedule = get_schedule_from_settings(settings)
    steps = list_sample_steps(samples_dir, min_step=min_step, max_step=max_step)
    if not steps:
        return pd.DataFrame(columns=["step", "reverse_step", *METRICS])
    x0 = np.load(dataset_files[0]).astype(np.float32)
    rng = np.random.default_rng(seed)
    records = []
    for s in steps:
        gen_path = samples_dir / f"samples_step_{s}.npy"
        if not gen_path.exists():
            continue
        xg = np.load(gen_path).astype(np.float32)
        # Compare against the clean real dataset (no diffusion)
        xr = x0
        # Subsample for PRDC if necessary
        if max_points is not None:
            n = min(max_points, len(xg), len(xr))
            idx_g = rng.choice(len(xg), size=n, replace=False)
            idx_r = rng.choice(len(xr), size=n, replace=False)
            xg = xg[idx_g]
            xr = xr[idx_r]
        m = compute_prdc(xr, xg, nearest_k=nearest_k)
        rec = {"step": s, "reverse_step": (num_steps - 1 - s)}
        rec.update(m)
        records.append(rec)
    if not records:
        return pd.DataFrame(columns=["step", "reverse_step", *METRICS])
    df = pd.DataFrame.from_records(records).sort_values("step").reset_index(drop=True)
    return df


# ------------------------------
# Plotting
# ------------------------------

def plot_reverse_step_lines(df: pd.DataFrame, title_prefix: str, outdir: Path, filename_suffix: str = "", reg: Optional[float] = None):
    if df.empty:
        return
    x = df["reverse_step"].values.astype(float)
    for m in METRICS:
        if m not in df.columns:
            continue
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, df[m].values, marker='o', linewidth=1.2)
        ax.set_xlabel("reverse step (T-1-step)")
        ax.set_ylabel(m)
        ax.set_title(f"{title_prefix} — {m.capitalize()}")
        # Stamp reg descriptor inside the plot for clarity
        if reg is not None:
            ax.text(0.02, 0.95, f"reg={reg:g}", transform=ax.transAxes, fontsize=10, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='none'))
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        suffix = filename_suffix
        if reg is not None:
            suffix = f"{suffix}_reg-{reg:g}"
        name = f"reverse_step_prdc_{m}{suffix}.png"
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
    k = max(2, int(np.ceil(max(0.01, min(1.0, frac)) * n)))
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


def reg_suffix(regs_filter: Optional[Iterable[float]]) -> str:
    if regs_filter is None:
        return ""
    regs = sorted(set(float(r) for r in regs_filter))
    return f"_regs-{'_'.join([str(r) for r in regs])}"


def plot_reverse_step_overlay(reg_to_df: Dict[float, pd.DataFrame], title_prefix: str, outdir: Path, filename_suffix: str = ""):
    if not reg_to_df:
        return
    regs = sorted(reg_to_df.keys())
    colors = plt.cm.tab10.colors
    for m in METRICS:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, reg in enumerate(regs):
            df = reg_to_df[reg]
            if df is None or df.empty or (m not in df.columns):
                continue
            x = df["reverse_step"].values.astype(float)
            y = df[m].values.astype(float)
            ax.plot(x, y, marker='o', linewidth=1.4, label=f"reg={reg:g}", color=colors[i % len(colors)])
        ax.set_xlabel("reverse step (T-1-step)")
        ax.set_ylabel(m)
        ax.set_title(f"{title_prefix} — {m.capitalize()} (overlay)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        name = f"reverse_step_prdc_{m}{filename_suffix}.png"
        fig.savefig(outdir / name, dpi=200)
        plt.close(fig)


def plot_reverse_step_overlay_smoothed_aggregated(run_to_reg_to_df: Dict[int, Dict[float, pd.DataFrame]], title_prefix: str, outdir: Path, filename_suffix: str = "", num_points: int = 400, frac: float = 0.3):
    if not run_to_reg_to_df:
        return
    # Aggregate points across runs per reg and metric
    reg_to_scatter: Dict[float, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for reg_map in run_to_reg_to_df.values():
        for reg, df in reg_map.items():
            for m in METRICS:
                if df is None or df.empty or (m not in df.columns):
                    continue
                x = df["reverse_step"].values.astype(float)
                y = df[m].values.astype(float)
                d = reg_to_scatter.setdefault(reg, {})
                xs, ys = d.get(m, (np.array([], dtype=float), np.array([], dtype=float)))
                d[m] = (np.concatenate([xs, x]), np.concatenate([ys, y]))
    regs = sorted(reg_to_scatter.keys())
    if not regs:
        return
    colors = plt.cm.tab10.colors
    for m in METRICS:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, reg in enumerate(regs):
            xs_all, ys_all = reg_to_scatter[reg].get(m, (None, None))
            if xs_all is None or xs_all.size == 0:
                continue
            gx = np.linspace(float(xs_all.min()), float(xs_all.max()), num_points)
            gy = _loess_fit(xs_all, ys_all, gx, frac=frac)
            ax.plot(gx, gy, linewidth=2.0, label=f"reg={reg:g}", color=colors[i % len(colors)])
        ax.set_xlabel("reverse step (T-1-step)")
        ax.set_ylabel(m)
        ax.set_title(f"{title_prefix} — {m.capitalize()} (smoothed aggregate)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        name = f"reverse_step_prdc_{m}_smoothed{filename_suffix}.png"
        fig.savefig(outdir / name, dpi=200)
        plt.close(fig)


def aggregate_mean_across_runs(run_to_df: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    bucket: Dict[int, Dict[str, List[float]]] = {}
    for df in run_to_df.values():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            rs = int(row["reverse_step"])
            entry = bucket.setdefault(rs, {m: [] for m in METRICS})
            for m in METRICS:
                entry[m].append(float(row[m]))
    rows = []
    for rs, mvals in bucket.items():
        r = {"reverse_step": rs}
        for m in METRICS:
            vals = mvals[m]
            if vals:
                r[m] = float(np.mean(vals))
        rows.append(r)
    if not rows:
        return pd.DataFrame(columns=["reverse_step", *METRICS])
    return pd.DataFrame(rows).sort_values("reverse_step").reset_index(drop=True)


# ------------------------------
# Orchestration
# ------------------------------

def process_objective(base_dir: Path, dataset: str, objective: str, runs: Iterable[int], regs_filter: Optional[Iterable[float]], nearest_k: int, max_points: Optional[int], min_step: Optional[int], max_step: Optional[int], recompute: bool = False):
    dataset_dir = base_dir / dataset
    run_to_reg_to_df: Dict[int, Dict[float, pd.DataFrame]] = {}
    for run_id in runs:
        run_dir = dataset_dir / f"{objective}_run_{run_id}"
        if not run_dir.exists():
            continue
        reg_to_df: Dict[float, pd.DataFrame] = {}
        for reg_dir in sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("reg_")]):
            reg_str = reg_dir.name.split("reg_")[1]
            try:
                reg_val = float(reg_str)
            except Exception:
                continue
            if regs_filter is not None and reg_val not in set(float(r) for r in regs_filter):
                continue
            cache_path = cached_prdc_csv_path(reg_dir, dataset, reg_str)
            if recompute or (not cache_path.exists()):
                df = compute_prdc_for_steps(reg_dir, dataset, reg_str, nearest_k=nearest_k, max_points=max_points, min_step=min_step, max_step=max_step)
                if not df.empty:
                    cache_path.write_text(df.to_csv(index=False))
            else:
                df = pd.read_csv(cache_path)
            if not df.empty:
                reg_to_df[reg_val] = df
        if reg_to_df:
            run_to_reg_to_df[run_id] = reg_to_df

    # Mean across runs: average metric at each reverse_step for each reg
    # Build per-reg mean df
    reg_to_mean_df: Dict[float, pd.DataFrame] = {}
    # organize by reg
    regs = sorted({reg for reg_map in run_to_reg_to_df.values() for reg in reg_map.keys()})
    for reg in regs:
        run_to_df = {run_id: reg_map.get(reg) for run_id, reg_map in run_to_reg_to_df.items()}
        mean_df = aggregate_mean_across_runs(run_to_df)
        if not mean_df.empty:
            reg_to_mean_df[reg] = mean_df
    if reg_to_mean_df:
        outdir_mean = ensure_outdir(base_dir, dataset, objective, "mean_run")
        # Save
        rows = []
        for reg, df in sorted(reg_to_mean_df.items()):
            tmp = df.copy()
            tmp.insert(0, "reg", reg)
            rows.append(tmp)
        df_long = pd.concat(rows, ignore_index=True)
        (outdir_mean / "summary_reverse_step_prdc.csv").write_text(df_long.to_csv(index=False))
        # Plots: overlay regs for mean across runs
        title_mean = f"{dataset} · {objective} · mean_run (reverse step)"
        suffix = reg_suffix(regs_filter)
        plot_reverse_step_overlay(reg_to_mean_df, title_prefix=title_mean, outdir=outdir_mean, filename_suffix=suffix)

        # Smoothed aggregate across all runs (single folder smooth_run)
        outdir_smooth = ensure_outdir(base_dir, dataset, objective, "smooth_run")
        plot_reverse_step_overlay_smoothed_aggregated(run_to_reg_to_df, title_prefix=f"{dataset} · {objective} · smooth_run (reverse step)", outdir=outdir_smooth, filename_suffix=suffix)


def main():
    ap = argparse.ArgumentParser(description="PRDC vs reverse step from saved samples.")
    ap.add_argument("--base_dir", type=str, default="outputs", help="Root folder that contains <dataset> directories.")
    ap.add_argument("--datasets", type=str, nargs="+", required=True, help="Datasets to process (names or 'all').")
    ap.add_argument("--objectives", type=str, nargs="*", default=None, help="Objectives to include (names or 'all'). If omitted, discover all present.")
    ap.add_argument("--runs", type=int, nargs="*", default=[1, 2, 3], help="Run IDs to include (default: 1 2 3).")
    ap.add_argument("--regs", type=float, nargs="*", default=None, help="Optional list of reg values to include (e.g., 0.0 0.3).")
    ap.add_argument("--nearest_k", type=int, default=None, help="Nearest-k for PRDC; default reads from settings or uses 5.")
    ap.add_argument("--max_points", type=int, default=3000, help="Max points per set for PRDC (subsampled if larger). Use None for all.")
    ap.add_argument("--min_step", type=int, default=None, help="Minimum step index to include.")
    ap.add_argument("--max_step", type=int, default=None, help="Maximum step index to include.")
    ap.add_argument("--recompute", action="store_true", help="Recompute PRDC per step and overwrite cached CSVs.")
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
            # Determine nearest_k: use CLI or first reg settings fallback
            nk = args.nearest_k
            if nk is None:
                # Try reading from run_1/reg_0.0 settings if exists
                guess_reg_dir = dataset_dir / f"{obj}_run_1" / "reg_0.0"
                if guess_reg_dir.exists():
                    st = read_settings(guess_reg_dir)
                    nk = int(st.get("nearest_k", 5))
                else:
                    nk = 5
            process_objective(base, ds, obj, args.runs, args.regs, nearest_k=nk, max_points=args.max_points if args.max_points and args.max_points > 0 else None, min_step=args.min_step, max_step=args.max_step, recompute=args.recompute)
            print(f"[OK] {ds}/{obj}: reverse-step PRDC plotted (mean and smooth runs).")


if __name__ == "__main__":
    main()
