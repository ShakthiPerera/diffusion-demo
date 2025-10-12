"""
Density visualization utility for diffusion experiment outputs.

For each dataset/objective/run/reg directory produced by training, this script
loads the saved real dataset (`dataset_*.npy`) and generated samples
(`generated_*_reg_*.npy`), estimates local density via k-nearest-neighbour
radii, and produces color-coded scatter plots:

  • Real density plot: original dataset coloured by estimated density.
  • Generated density plot: generated samples coloured by density estimated
    against the real dataset (small radius ⇒ high density).

Outputs are saved under:
  results/density_plots/<dataset>/<objective>/mean_run/reg_<reg>/
      real_density.png
      generated_density.png
      real_density.csv            (x, y, radius, density)
      generated_density.csv       (x, y, radius, density)

The script mirrors the CLI conventions of other helpers in this repository, so
you can target specific datasets/objectives/runs/regs or use the "all"
shorthand to process everything available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.neighbors import NearestNeighbors


CATEGORY_LABELS = ["min", "mid-low", "mid-high", "max"]
PALETTE_FRACTIONS = [0.10, 0.40, 0.80, 0.90]
PALETTE_CHOICES = [
    "coolwarm",
    "viridis",
    "plasma",
    "magma",
    "cividis",
    "Spectral",
    "RdYlBu",
    "PuOr",
]


def _cmap_hex(cmap_name: str, fraction: float) -> str:
    cmap = plt.get_cmap(cmap_name)
    r, g, b, _ = cmap(max(0.0, min(1.0, fraction)))
    return "#%02x%02x%02x" % (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def build_palette(palette_name: str, fractions: List[float]) -> List[str]:
    return [_cmap_hex(palette_name, f) for f in fractions]


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def discover_datasets(base_dir: Path) -> List[str]:
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir() and p.name != "results"])


def discover_objectives(dataset_dir: Path) -> List[str]:
    objs = set()
    for p in dataset_dir.iterdir():
        if p.is_dir() and "_run_" in p.name:
            objs.add(p.name.split("_run_")[0])
    return sorted(objs)


# ---------------------------------------------------------------------------
# Density helpers
# ---------------------------------------------------------------------------

def knn_radii(
    base_points: np.ndarray,
    query_points: np.ndarray,
    k: int,
    leave_one_out: bool = False,
) -> np.ndarray:
    """Return k-NN radius (distance to k-th neighbour) for each query point.

    When `leave_one_out=True`, each query point is assumed to be present in the
    base set (e.g., real-vs-real density); we therefore ask for k+1 neighbours
    so that the k-th *other* neighbour is returned (the 0-distance self match is
    discarded by picking the last distance).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    n_neighbors = k + 1 if leave_one_out else k
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(base_points)
    dists, _ = nbrs.kneighbors(query_points)
    radii = dists[:, -1]
    return radii


def estimate_density_from_radius(radii: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert radii to a simple density proxy (inverse radius)."""
    radii = np.asarray(radii, dtype=np.float64)
    return 1.0 / (radii + eps)


def maybe_subsample(arr: np.ndarray, max_points: Optional[int], rng: np.random.Generator) -> np.ndarray:
    if max_points is None or len(arr) <= max_points:
        return arr
    idx = rng.choice(len(arr), size=max_points, replace=False)
    return arr[idx]


def ensure_outdir(base_dir: Path, dataset: str, objective: str, subpath: str) -> Path:
    out = base_dir / "results" / "density_plots" / dataset / objective / subpath
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_density_csv(outdir: Path, filename: str, points: np.ndarray, radii: np.ndarray, densities: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "radius": radii,
            "density": densities,
        }
    )
    (outdir / filename).write_text(df.to_csv(index=False))


def build_density_bins(
    densities: np.ndarray,
    category_bounds: Optional[List[float]],
    category_percentiles: Optional[List[float]],
) -> np.ndarray:
    finite = densities[np.isfinite(densities)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=float)
    finite = finite.astype(float)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        hi = lo + 1e-8

    if category_bounds:
        thresholds = sorted(float(b) for b in category_bounds)
    else:
        if category_percentiles:
            perc = sorted(float(p) for p in category_percentiles)
        else:
            perc = [40.0, 70.0, 90.0]
        thresholds = [float(np.percentile(finite, np.clip(p, 0.0, 100.0))) for p in perc]

    edges = [lo]
    for t in thresholds:
        edges.append(max(edges[-1], t))
    edges.append(max(edges[-1] + 1e-8, hi))
    return np.array(edges, dtype=float)


def plot_density_scatter(
    points: np.ndarray,
    densities: np.ndarray,
    title: str,
    color_label: str,
    out_path: Path,
    bins: np.ndarray,
    palette_colors: List[str],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = ListedColormap(palette_colors)
    norm = BoundaryNorm(bins, cmap.N, clip=True)
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=densities,
        cmap=cmap,
        norm=norm,
        s=16,
        linewidths=0.0,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    tick_positions = [(bins[i] + bins[i + 1]) / 2.0 for i in range(len(bins) - 1)]
    cbar = fig.colorbar(sc, ax=ax, boundaries=bins, ticks=tick_positions)
    cbar.ax.set_yticklabels(CATEGORY_LABELS)
    cbar.set_label(color_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def load_dataset_and_generated(reg_dir: Path, dataset_name: str, reg_str: str) -> Tuple[np.ndarray, np.ndarray]:
    dataset_files = sorted(reg_dir.glob(f"dataset_{dataset_name}.npy"))
    if not dataset_files:
        raise FileNotFoundError(f"Real dataset file not found in {reg_dir}")
    real = np.load(dataset_files[0]).astype(np.float32)

    gen_pattern = f"generated_{dataset_name}_reg_{reg_str}.npy"
    generated_files = sorted(reg_dir.glob(gen_pattern))
    if not generated_files:
        raise FileNotFoundError(f"Generated samples file not found (expected {gen_pattern})")
    generated = np.load(generated_files[0]).astype(np.float32)
    return real, generated


def process_reg(
    base_dir: Path,
    dataset: str,
    objective: str,
    reg_str: str,
    reg_val: float,
    reg_dirs: List[Path],
    nearest_k: int,
    max_points: Optional[int],
    rng: np.random.Generator,
    category_bounds: Optional[List[float]],
    category_percentiles: Optional[List[float]],
    palette_colors: List[str],
) -> None:
    real_arrays: List[np.ndarray] = []
    gen_arrays: List[np.ndarray] = []
    for reg_dir in reg_dirs:
        try:
            real, generated = load_dataset_and_generated(reg_dir, dataset, reg_str)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")
            continue
        if real.ndim != 2 or real.shape[1] != 2:
            print(f"[WARN] Skipping {reg_dir}: expected 2D real points, got shape {real.shape}")
            continue
        if generated.ndim != 2 or generated.shape[1] != 2:
            print(f"[WARN] Skipping {reg_dir}: expected 2D generated points, got shape {generated.shape}")
            continue
        real_arrays.append(real)
        gen_arrays.append(generated)

    if not real_arrays or not gen_arrays:
        return

    real_concat = np.concatenate(real_arrays, axis=0)
    gen_concat = np.concatenate(gen_arrays, axis=0)

    real_sample = maybe_subsample(real_concat, max_points, rng)
    gen_sample = maybe_subsample(gen_concat, max_points, rng)

    real_radii = knn_radii(real_sample, real_sample, nearest_k, leave_one_out=True)
    real_density = estimate_density_from_radius(real_radii)

    gen_radii = knn_radii(real_sample, gen_sample, nearest_k, leave_one_out=False)
    gen_density = estimate_density_from_radius(gen_radii)

    combined_density = np.concatenate([real_density, gen_density])
    bins = build_density_bins(combined_density, category_bounds, category_percentiles)

    outdir = ensure_outdir(base_dir, dataset, objective, f"mean_run/reg_{reg_str}")
    save_density_csv(outdir, "real_density.csv", real_sample, real_radii, real_density)
    save_density_csv(outdir, "generated_density.csv", gen_sample, gen_radii, gen_density)

    title_suffix = f"reg {reg_val:g}"
    plot_density_scatter(
        real_sample,
        real_density,
        title=f"real · {title_suffix}",
        color_label="k-NN density (1 / radius)",
        out_path=outdir / "real_density.png",
        bins=bins,
        palette_colors=palette_colors,
    )

    plot_density_scatter(
        gen_sample,
        gen_density,
        title=f"generated · {title_suffix}",
        color_label="k-NN density vs real (1 / radius)",
        out_path=outdir / "generated_density.png",
        bins=bins,
        palette_colors=palette_colors,
    )


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Density plots for real vs generated samples (colour-coded by k-NN density).")
    ap.add_argument("--base_dir", type=str, default="outputs", help="Root directory containing <dataset> folders (default: outputs).")
    ap.add_argument("--datasets", type=str, nargs="+", required=True, help="Datasets to process (names or 'all').")
    ap.add_argument("--objectives", type=str, nargs="*", default=None, help="Objectives to include (names or 'all'). If omitted, discover all present.")
    ap.add_argument("--runs", type=int, nargs="*", default=[1, 2, 3], help="Run IDs to include (default: 1 2 3).")
    ap.add_argument("--regs", type=float, nargs="*", default=None, help="Regularisation values to include (e.g., 0.0 0.3). If omitted, use all found.")
    ap.add_argument("--nearest_k", type=int, default=10, help="k for k-NN density estimation (default: 10).")
    ap.add_argument("--max_points", type=int, default=5000, help="Optional cap on points per plot to keep processing light. Use 0 for all points.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for subsampling when --max_points is used.")
    ap.add_argument(
        "--category_bounds",
        type=float,
        nargs="*",
        default=None,
        help="Explicit density thresholds (three numbers) separating low → high categories.",
    )
    ap.add_argument(
        "--category_percentiles",
        type=float,
        nargs="*",
        default=None,
        help="Percentiles (0-100) used to derive category thresholds; default: 40 70 90.",
    )
    ap.add_argument(
        "--palette",
        type=str,
        default="coolwarm",
        choices=PALETTE_CHOICES,
        help="Matplotlib palette to sample for the four density categories (default: coolwarm).",
    )
    ap.add_argument(
        "--list_palettes",
        action="store_true",
        help="List available palettes with sampled colours (using the configured fractions) and exit.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Base dir not found: {base}")

    if args.nearest_k <= 0:
        raise ValueError("--nearest_k must be positive")

    if args.category_bounds is not None and len(args.category_bounds) != 3:
        raise ValueError("--category_bounds expects exactly 3 values (low→high thresholds)")
    if args.category_percentiles is not None and len(args.category_percentiles) != 3:
        raise ValueError("--category_percentiles expects exactly 3 values (for 4 categories)")

    if args.list_palettes:
        print("Available palettes (fractions: {}):".format(
            ", ".join(f"{int(f * 100)}%" for f in PALETTE_FRACTIONS)
        ))
        for name in PALETTE_CHOICES:
            colors = build_palette(name, PALETTE_FRACTIONS)
            print(f"  {name:>8}: {', '.join(colors)}")
        return

    palette_colors = build_palette(args.palette, PALETTE_FRACTIONS)

    max_points = None if args.max_points is None or args.max_points <= 0 else args.max_points
    rng = np.random.default_rng(args.seed)

    if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
        datasets = discover_datasets(base)
    else:
        datasets = list(args.datasets)

    for dataset in datasets:
        dataset_dir = base / dataset
        if not dataset_dir.exists():
            print(f"[WARN] Dataset not found: {dataset_dir}")
            continue

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

        for objective in objectives:
            reg_to_dirs: Dict[str, Dict[str, object]] = {}
            for run_id in args.runs:
                run_dir = dataset_dir / f"{objective}_run_{run_id}"
                if not run_dir.exists():
                    print(f"[WARN] Missing run directory: {run_dir}")
                    continue
                for reg_path in run_dir.iterdir():
                    if not (reg_path.is_dir() and reg_path.name.startswith("reg_")):
                        continue
                    reg_str = reg_path.name.split("reg_")[1]
                    try:
                        reg_val = float(reg_str)
                    except Exception:
                        continue
                    if args.regs is not None and not any(abs(reg_val - r) < 1e-9 for r in args.regs):
                        continue
                    entry = reg_to_dirs.setdefault(reg_str, {"value": reg_val, "paths": []})
                    entry["paths"].append(reg_path)

            if not reg_to_dirs:
                print(f"[WARN] No matching reg directories found for {dataset}/{objective}")
                continue

            for reg_str, info in sorted(reg_to_dirs.items(), key=lambda item: item[1]["value"]):
                reg_val = info["value"]
                reg_dirs = info["paths"]
                try:
                    process_reg(
                        base,
                        dataset,
                        objective,
                        reg_str,
                        reg_val,
                        reg_dirs,
                        args.nearest_k,
                        max_points,
                        rng,
                        args.category_bounds,
                        args.category_percentiles,
                        palette_colors,
                    )
                    print(f"[OK] {dataset}/{objective}/mean_run/reg_{reg_val:g}: density plots saved.")
                except Exception as exc:
                    print(f"[ERROR] Failed for {dataset}/{objective}/mean_run/reg_{reg_val:g}: {exc}")


if __name__ == "__main__":
    main()
