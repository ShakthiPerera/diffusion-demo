import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class DatasetInfo:
    path: Path
    dataset: str  # e.g., central_banana
    objective: Optional[str]  # e.g., iso, mean_l2
    run: Optional[int]
    reg: Optional[float]


def anisotropy_2d(X: np.ndarray, robust: bool = False, eps: float = 1e-12) -> Dict[str, float]:
    """Compute 2D anisotropy metrics and ellipse parameters."""

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        X = X.reshape(len(X), -1)
        if X.shape[1] != 2:
            raise ValueError("anisotropy_2d expects 2D data")

    Xc = X - X.mean(axis=0, keepdims=True)
    if robust:
        s = np.std(Xc, axis=0, ddof=1) + eps
        z = Xc / s
        w = 1.0 / np.maximum(1.0, (np.sum(z ** 2, axis=1) / 2.0))
        W = (w / (w.mean() + eps))[:, None]
        Xc = Xc * W

    Sigma = np.cov(Xc, rowvar=False)
    lam, V = np.linalg.eigh(Sigma)
    lam = np.sort(np.maximum(lam, 0.0))[::-1]
    lam1, lam2 = lam
    if lam1 <= eps:
        return {
            "H_norm_bits": 0.0,
            "SF_2d": 0.0,
            "deff_frac": 0.0,
            "kappa": float("inf"),
            "eccentricity": 0.0,
            "theta": 0.0,
        }

    p = lam / (lam.sum() + eps)
    H = -(p * np.log(p + eps)).sum()
    H_norm_bits = H / np.log(2.0)
    SF = np.sqrt(lam1 * lam2) / ((lam1 + lam2) * 0.5 + eps)
    deff_pr = 1.0 / (p @ p)
    deff_frac = deff_pr / 2.0
    kappa = lam1 / (lam2 + eps)
    eccentricity = np.sqrt(1.0 - (lam2 / (lam1 + eps)))
    v1 = V[:, 1]
    theta = float(np.arctan2(v1[1], v1[0]))

    return {
        "H_norm_bits": float(H_norm_bits),
        "SF_2d": float(SF),
        "deff_frac": float(deff_frac),
        "kappa": float(kappa),
        "eccentricity": float(eccentricity),
        "theta": theta,
    }


def parse_path_info(path: Path) -> DatasetInfo:
    """Parse dataset, objective, run, and reg from a path like:
    outputs/central_banana/iso_run_2/reg_0.0/dataset_central_banana.npy

    This is best-effort; missing parts become None.
    """
    # Expect structure: outputs/<dataset>/<objective>_run_<N>/reg_<VAL>/dataset_*.npy
    parts = path.parts
    dataset = None
    objective = None
    run: Optional[int] = None
    reg: Optional[float] = None

    # Find "outputs" index and walk following parts for dataset hierarchy
    try:
        out_idx = parts.index("outputs")
    except ValueError:
        out_idx = -1

    if out_idx >= 0 and out_idx + 1 < len(parts):
        dataset = parts[out_idx + 1]

    # Try to parse objective_run from the next component
    if out_idx >= 0 and out_idx + 2 < len(parts):
        m = re.match(r"(.+)_run_(\d+)$", parts[out_idx + 2])
        if m:
            objective = m.group(1)
            try:
                run = int(m.group(2))
            except Exception:
                run = None

    # Try to parse reg from the next component
    if out_idx >= 0 and out_idx + 3 < len(parts):
        m = re.match(r"reg_([-+]?\d*\.?\d+)$", parts[out_idx + 3])
        if m:
            try:
                reg = float(m.group(1))
            except Exception:
                reg = None

    return DatasetInfo(path=path, dataset=(dataset or ""), objective=objective, run=run, reg=reg)


def compute_eigvals_from_data(X: np.ndarray) -> np.ndarray:
    """Return eigenvalues of the sample covariance matrix via SVD.

    - Centers X per-column
    - Uses singular values: eigvals = s^2 / (n-1)
    - Clips tiny negatives to zero and removes zeros
    """
    if X.ndim != 2:
        X = X.reshape(len(X), -1)
    Xc = X - X.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    eigvals = (s.astype(np.float64) ** 2) / max(n - 1, 1)
    eigvals = np.clip(eigvals, 0.0, None)
    eigvals = eigvals[eigvals > 0]
    return eigvals


def anisotropy_metrics(eigvals: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    if eigvals.size == 0 or float(eigvals.sum()) <= eps:
        return {
            "entropy": 0.0,
            "entropy_norm": 0.0,
            "spectral_flatness": 0.0,
        }
    lam_sum = float(eigvals.sum())
    p = eigvals / lam_sum
    H = float(-(p * np.log(p + eps)).sum())
    H_norm = float(H / (math.log(len(p)) if len(p) > 1 else 1.0))
    gmean = float(np.exp(np.log(eigvals + eps).mean()))
    amean = float(eigvals.mean())
    spectral_flatness = float(gmean / (amean + eps))
    return {
        "entropy": round(H, 6),
        "entropy_norm": round(H_norm, 6),
        "spectral_flatness": round(spectral_flatness, 6),
    }


def find_dataset_files(root: Path, pattern: str = "dataset_*.npy") -> List[Path]:
    return sorted(root.rglob(pattern))


def process_file(path: Path, robust: bool = False) -> Optional[Tuple[DatasetInfo, int, int, Dict[str, float]]]:
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception:
        # Some files may require allow_pickle
        try:
            arr = np.load(path, allow_pickle=True)
        except Exception:
            return None

    if arr.ndim == 1:
        # Could be a single array object holding the data
        try:
            arr = np.array(arr.tolist())
        except Exception:
            arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        arr = arr.reshape(len(arr), -1)

    arr = arr.astype(np.float64)
    n, d = int(arr.shape[0]), int(arr.shape[1])
    info = parse_path_info(path)
    eigvals = compute_eigvals_from_data(arr)
    base_metrics = anisotropy_metrics(eigvals)
    try:
        X2d = arr[:, :2]
        extra_metrics = anisotropy_2d(X2d, robust=robust)
    except Exception:
        extra_metrics = {
            "H_norm_bits": "",
            "SF_2d": "",
            "deff_frac": "",
            "kappa": "",
            "eccentricity": "",
            "theta": "",
        }

    metrics = {**base_metrics, **extra_metrics}
    return info, n, d, metrics


def write_csv(rows: Iterable[Tuple[DatasetInfo, int, int, Dict[str, float]]], out_path: Path, include_meta: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if include_meta:
        fieldnames = [
            "path",
            "dataset",
            "objective",
            "run",
            "reg",
            "n_samples",
            "dim",
            "entropy",
            "entropy_norm",
            "spectral_flatness",
            "H_norm_bits",
            "SF_2d",
            "deff_frac",
            "kappa",
            "eccentricity",
            "theta",
        ]
    else:
        fieldnames = [
            "dataset",
            "n_samples",
            "dim",
            "entropy",
            "entropy_norm",
            "spectral_flatness",
            "H_norm_bits",
            "SF_2d",
            "deff_frac",
            "kappa",
            "eccentricity",
            "theta",
        ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for info, n, d, metrics in rows:
            if include_meta:
                row = {
                    "path": str(info.path),
                    "dataset": info.dataset,
                    "objective": info.objective if info.objective is not None else "",
                    "run": info.run if info.run is not None else "",
                    "reg": info.reg if info.reg is not None else "",
                    "n_samples": n,
                    "dim": d,
                    "entropy": metrics.get("entropy", ""),
                    "entropy_norm": metrics.get("entropy_norm", ""),
                    "spectral_flatness": metrics.get("spectral_flatness", ""),
                    "H_norm_bits": metrics.get("H_norm_bits", ""),
                    "SF_2d": metrics.get("SF_2d", ""),
                    "deff_frac": metrics.get("deff_frac", ""),
                    "kappa": metrics.get("kappa", ""),
                    "eccentricity": metrics.get("eccentricity", ""),
                    "theta": metrics.get("theta", ""),
                }
            else:
                row = {
                    "dataset": info.dataset,
                    "n_samples": n,
                    "dim": d,
                    "entropy": metrics.get("entropy", ""),
                    "entropy_norm": metrics.get("entropy_norm", ""),
                    "spectral_flatness": metrics.get("spectral_flatness", ""),
                    "H_norm_bits": metrics.get("H_norm_bits", ""),
                    "SF_2d": metrics.get("SF_2d", ""),
                    "deff_frac": metrics.get("deff_frac", ""),
                    "kappa": metrics.get("kappa", ""),
                    "eccentricity": metrics.get("eccentricity", ""),
                    "theta": metrics.get("theta", ""),
                }
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Compute anisotropy metrics for 2D datasets stored as .npy arrays.")
    ap.add_argument("--root", type=str, default="outputs", help="Root directory to search for dataset_*.npy")
    ap.add_argument("--pattern", type=str, default="dataset_*.npy", help="Filename pattern to match")
    ap.add_argument("--out", type=str, default="outputs/results/anisotropy_2d.csv", help="Output CSV path")
    ap.add_argument("--include-meta", action="store_true", help="Include path/objective/run/reg columns in the CSV")
    ap.add_argument("--datasets", type=str, nargs="*", default=None, help="Optional list of dataset names to include (e.g. central_banana moon_scatter moon_circles swiss_roll)")
    ap.add_argument("--objective", type=str, default=None, help="Optional objective filter (e.g. iso, mean_l2)")
    ap.add_argument("--run", type=int, default=None, help="Optional run filter (e.g. 1)")
    ap.add_argument("--reg", type=float, default=None, help="Optional reg filter (e.g. 0.0)")
    ap.add_argument("--robust", action="store_true", help="Use robust weighting when computing 2D metrics")
    args = ap.parse_args()

    root = Path(args.root)
    files = find_dataset_files(root, args.pattern)
    results: List[Tuple[DatasetInfo, int, int, Dict[str, float]]] = []
    # Normalize dataset filter to a set for fast lookup
    dataset_allow: Optional[set] = None
    if args.datasets:
        dataset_allow = set(args.datasets)

    for p in files:
        info = parse_path_info(p)
        # Apply optional filters early
        if dataset_allow is not None and info.dataset not in dataset_allow:
            continue
        if args.objective is not None and info.objective != args.objective:
            continue
        if args.run is not None and info.run != args.run:
            continue
        if args.reg is not None:
            if info.reg is None:
                continue
            if not (abs(info.reg - float(args.reg)) < 1e-12):
                continue

        r = process_file(p, robust=args.robust)
        if r is not None:
            results.append(r)

    write_csv(results, Path(args.out), include_meta=args.include_meta)
    print(f"Processed {len(results)} datasets. Wrote {args.out}")


if __name__ == "__main__":
    main()
