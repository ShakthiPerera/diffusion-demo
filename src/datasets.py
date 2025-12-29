"""Lightweight synthetic datasets used for 2D diffusion experiments.

Only a handful of shapes are kept to simplify the project:

- moon_scatter: one moon plus a band of scattered points
- swiss_roll: 2D projection of the Swiss roll
- central_banana: banana tail with a dense centre
- moon_circles: one moon with two compact circles

Each dataset is normalised to [-1, 1] per dimension.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons, make_swiss_roll


class Synthetic2DDataset:
    """Generate synthetic 2D datasets with a unified interface."""

    SUPPORTED = ("moon_scatter", "swiss_roll", "central_banana", "moon_circles")

    def __init__(self, name: str, num_samples: int = 10_000, noise_level: float = 0.1, random_state: int = 42) -> None:
        self.name = name.lower()
        if self.name == "moon_scattering":
            self.name = "moon_scatter"
        if self.name not in self.SUPPORTED:
            supported = ", ".join(self.SUPPORTED)
            raise ValueError(f"Unknown dataset '{name}'. Supported: {supported}")
        self.num_samples = int(num_samples)
        self.noise_level = float(noise_level)
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)

    # ------------------------------------------------------------------
    def generate(self) -> np.ndarray:
        if self.name == "moon_scatter":
            data = self._generate_moon_scatter()
        elif self.name == "swiss_roll":
            data = self._generate_swiss_roll()
        elif self.name == "central_banana":
            data = self._generate_central_banana()
        else:  # moon_circles
            data = self._generate_moon_circles()
        return self._normalize(data)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        min_val = X.min(axis=0, keepdims=True)
        max_val = X.max(axis=0, keepdims=True)
        scale = np.where(max_val > min_val, max_val - min_val, 1.0)
        X_norm = 2.0 * (X - min_val) / scale - 1.0
        return X_norm.astype(np.float32)

    # ------------------------------------------------------------------
    def _generate_moon_scatter(self) -> np.ndarray:
        max_scatter = min(500, self.num_samples // 2)
        total_moon_points = max(2 * self.num_samples - 2 * max_scatter, 2)
        X_full, y_full = make_moons(n_samples=total_moon_points, noise=0.1, random_state=self.random_state)
        crescent = X_full[y_full == 0]
        required_crescent = self.num_samples - max_scatter
        if crescent.shape[0] >= required_crescent:
            crescent = crescent[:required_crescent]
        else:
            repeats = (required_crescent + crescent.shape[0] - 1) // max(crescent.shape[0], 1)
            crescent = np.tile(crescent, (repeats, 1))[:required_crescent]
        noise = 0.3 * self.rng.standard_normal((crescent.shape[0], 2))
        scatter_points = crescent[:max_scatter] + noise[:max_scatter]
        return np.vstack((crescent, scatter_points)).astype(np.float64)

    def _generate_swiss_roll(self) -> np.ndarray:
        X, _ = make_swiss_roll(n_samples=self.num_samples, noise=self.noise_level, random_state=self.random_state)
        return X[:, [0, 2]].astype(np.float64)

    def _generate_central_banana(self) -> np.ndarray:
        default_central = min(500, max(0, self.num_samples))
        central_points = min(default_central, self.num_samples)
        tail_points = max(0, self.num_samples - central_points)
        sigma = 0.3
        max_distance = 1.0
        concentration_factor = 2.0
        decay = 0.6

        def banana_pdf(theta: float) -> float:
            return max_distance * (np.exp(-decay * theta) * (1.0 + concentration_factor * np.sin(theta)))

        tail = []
        for _ in range(tail_points):
            theta = self.rng.exponential(scale=0.25)
            r = banana_pdf(theta)
            width = sigma * (1.0 - (r / (2.0 * max_distance)))
            x_center = r * np.cos(theta)
            y_center = r * np.sin(theta)
            x_offset = self.rng.uniform(-width / 2.0, width / 2.0)
            y_offset = self.rng.uniform(-width / 2.0, width / 2.0)
            tail.append((x_center + x_offset, y_center + y_offset))
        tail = np.array(tail, dtype=np.float64) if tail else np.empty((0, 2), dtype=np.float64)

        def central_point() -> tuple[float, float]:
            theta = np.pi / 3.0 + self.rng.normal(loc=0.0, scale=0.1)
            r = max_distance * self.rng.uniform(0.5, 0.7)
            x_center = r * np.cos(theta)
            y_center = r * np.sin(theta)
            x_offset = self.rng.normal(loc=-0.5, scale=sigma / 5.0)
            y_offset = self.rng.normal(loc=-0.25, scale=sigma / 5.0)
            return (x_center + x_offset, y_center + y_offset)

        center = np.array([central_point() for _ in range(central_points)], dtype=np.float64)
        return np.vstack((tail, center)) if tail_points > 0 else center

    def _generate_moon_circles(self) -> np.ndarray:
        circle_density = min(500, max(0, self.num_samples // 3))
        crescent_points = max(0, self.num_samples - 2 * circle_density)
        total_moon_points = max(0, 2 * self.num_samples - 4 * circle_density)
        total_moon_points = max(total_moon_points, 2 * crescent_points)
        X_full, y_full = make_moons(n_samples=total_moon_points, noise=self.noise_level, random_state=self.random_state)
        crescent = X_full[y_full == 0]
        if crescent.shape[0] >= crescent_points:
            crescent = crescent[:crescent_points]
        else:
            repeats = (crescent_points + crescent.shape[0] - 1) // max(crescent.shape[0], 1)
            crescent = np.tile(crescent, (repeats, 1))[:crescent_points]
        crescent = crescent.astype(np.float64)

        circle_radius = 0.2
        center1 = np.array([-1.0, -0.75], dtype=np.float64)
        center2 = np.array([1.0, -0.75], dtype=np.float64)
        angles1 = 2.0 * np.pi * self.rng.random(circle_density)
        angles2 = 2.0 * np.pi * self.rng.random(circle_density)
        radii1 = circle_radius * np.sqrt(self.rng.random(circle_density)) + 0.05 * self.rng.standard_normal(circle_density)
        radii2 = circle_radius * np.sqrt(self.rng.random(circle_density)) + 0.05 * self.rng.standard_normal(circle_density)
        circle1 = np.c_[radii1 * np.cos(angles1) + center1[0], radii1 * np.sin(angles1) + center1[1]]
        circle2 = np.c_[radii2 * np.cos(angles2) + center2[0], radii2 * np.sin(angles2) + center2[1]]
        return np.vstack((crescent, circle1, circle2)).astype(np.float64)


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Visualize all synthetic 2D datasets.")
    parser.add_argument("--samples", type=int, default=10_000, help="Number of points per dataset.")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level for noisy datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--marker-size", type=float, default=4.0, help="Scatter marker size.")
    parser.add_argument("--alpha", type=float, default=0.6, help="Scatter alpha.")
    parser.add_argument("--output", type=str, default=None, help="Output image path.")
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    args = parser.parse_args()

    dataset_names = Synthetic2DDataset.SUPPORTED
    num_datasets = len(dataset_names)
    cols = 2
    rows = (num_datasets + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)

    for ax, name in zip(axes.ravel(), dataset_names):
        dataset = Synthetic2DDataset(
            name=name,
            num_samples=args.samples,
            noise_level=args.noise,
            random_state=args.seed,
        )
        points = dataset.generate()
        ax.scatter(points[:, 0], points[:, 1], s=args.marker_size, alpha=args.alpha)
        ax.set_title(name)
        ax.set_aspect("equal", "box")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused axes in the grid.
    for ax in axes.ravel()[num_datasets:]:
        ax.axis("off")

    fig.tight_layout()
    output_path = Path(args.output) if args.output else Path(__file__).with_name("synthetic_datasets.png")
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    if args.show:
        plt.show()
