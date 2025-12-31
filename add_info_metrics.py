"""Populate KL and mutual information metrics into existing entropy JSON files.

This walks every ``true_*.npy`` under ``outputs/`` (or a custom root), computes:
  - KL(p_data || fitted Gaussian) via kNN
  - Mutual information (Kraskov kNN)
using helpers from ``src/information_utils.py`` and writes them into the matching
``entropy_<dataset>.json`` files alongside the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from src import information_utils as ent


def update_entropy_files(root: Path, k: int) -> list[Path]:
    """Compute metrics for each true_*.npy and update its entropy JSON."""
    updated: list[Path] = []
    true_files = sorted(root.rglob("true_*.npy"))
    if not true_files:
        raise FileNotFoundError(f"No true_*.npy files found under {root}.")

    for true_path in true_files:
        data = np.load(true_path)
        kl = ent.kl_data_to_gaussian_knn(data, k=k)
        mi = ent.mutual_information_knn(data, k=k)

        dataset = true_path.stem
        if dataset.startswith("true_"):
            dataset = dataset[len("true_") :]

        json_path = true_path.with_name(f"entropy_{dataset}.json")
        if not json_path.exists():
            print(f"Skipping {true_path}: missing {json_path}")
            continue

        with open(json_path, "r") as f:
            payload = json.load(f)

        payload["true_kl_to_gaussian_knn"] = kl
        payload["true_mutual_information_knn"] = mi

        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        updated.append(json_path)
        print(f"Updated {json_path} (KL={kl:.6f}, MI={mi:.6f})")

    return updated


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute KL and mutual information for all true_*.npy under outputs/."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing run folders.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k for the kNN-based estimators.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    updated = update_entropy_files(args.root, k=args.k)
    if not updated:
        print("No entropy files updated.")


if __name__ == "__main__":
    main()
