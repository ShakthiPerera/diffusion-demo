import argparse
import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm

from datasets import (
    CentralBananaDataset, MoonWithScatteringsDataset, MoonWithTwoCiclesBoundedDataset,
    MoonWithTwoCirclesUnboundedDataset, SwissRollDataset, GMMDataset
)
from diffusion import ConditionalDenseModel
from functions import make_beta_schedule
from diffusion.ddpm_norms import DDPM as ddpm


# -----------------------------
# Density utilities (k-NN radii + discrete bins)
# -----------------------------
def knn_radii(real_features: np.ndarray, query_features: np.ndarray, k: int, leave_one_out: bool = False) -> np.ndarray:
    n_neighbors = k + 1 if leave_one_out else k
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(real_features)
    dists, _ = nbrs.kneighbors(query_features)
    return dists[:, -1]


def get_asymmetric_bins_radii(radii: np.ndarray) -> np.ndarray:
    """Create dynamic bins for radii based on percentiles, consistent with counts binning."""
    lo, hi = float(np.min(radii)), float(np.max(radii))
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    # Use percentiles to create a dynamic spread
    percentiles = np.percentile(radii, [80, 90, 97])
    bins = [lo, percentiles[0], percentiles[1], percentiles[2], hi]
    bins = np.maximum.accumulate(bins)
    if len(np.unique(bins)) < 5:
        bins = np.linspace(lo, hi, 5)
    return np.array(bins, dtype=float)


def get_asymmetric_bins_counts(counts: np.ndarray) -> np.ndarray:
    """Create dynamic bins for point counts based on percentiles, inspired by [0, 1, 4, 6, max_count]."""
    lo, hi = 0, int(np.max(counts))
    # if hi <= lo:
    #     hi = lo + 1
    # # Use percentiles to create a dynamic spread
    # percentiles = np.percentile(counts, [0.1, 0.7, 2])
    bins = [lo, int(lo + 0.5*(hi-lo)), int(lo + 0.1*(hi-lo)),int(lo + 0.2*(hi-lo)), hi]
    bins = np.maximum.accumulate(bins)
    if len(np.unique(bins)) < 5:
        bins = np.linspace(lo, hi, 5)
    return np.array(bins, dtype=float)


def count_points_within_radius(real_features: np.ndarray, fake_features: np.ndarray, nearest_k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    For each point in the fake dataset, calculate the count of real points within the k-NN radius
    determined by distances within the real dataset.

    Args:
        real_features (np.ndarray): Real data features, shape [N, feature_dim].
        fake_features (np.ndarray): Fake/generated data features, shape [M, feature_dim].
        nearest_k (int): Number of nearest neighbors to define the radius.

    Returns:
        tuple[np.ndarray, np.ndarray]: Counts of real points within radius for each fake point,
                                      and k-th nearest neighbor radii for real points.
    """
    # Compute k-th nearest neighbor distances for real points (leave-one-out)
    real_nbrs = NearestNeighbors(n_neighbors=nearest_k + 1, n_jobs=-1).fit(real_features)
    real_dists, _ = real_nbrs.kneighbors(real_features)
    kth_nearest_radius = real_dists[:, -1]  # k-th nearest neighbor distance

    # Compute distances from fake points to all real points
    fake_nbrs = NearestNeighbors(n_neighbors=len(real_features), n_jobs=-1).fit(real_features)
    dists, _ = fake_nbrs.kneighbors(fake_features)
    
    # Count real points within the k-th nearest neighbor radius for each fake point
    counts = np.sum(dists <= kth_nearest_radius[None, :], axis=1)
    return counts, kth_nearest_radius


def plot_color_coded_points(data: np.ndarray, values: np.ndarray, save_path: str, is_generated: bool = False) -> None:
    """
    Plot color-coded scatter points for training (radii) or generated (counts) data.

    Args:
        data (np.ndarray): Data points to plot, shape [N, 2].
        values (np.ndarray): Radii or counts for color mapping.
        save_path (str): Path to save the plot.
        is_generated (bool): If True, plot generated data with count-based bins; else, plot training data with radii-based bins.
    """
    if is_generated:
        bins = get_asymmetric_bins_counts(values)
        colors = ['blue', 'lightblue', 'lightcoral', 'red']
        label = 'Points within k-NN radius'
    else:
        bins = get_asymmetric_bins_radii(values)
        colors = ['red', 'lightcoral', 'lightblue', 'blue']  # Reversed for training data
        label = 'k-NN radius'

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bins, cmap.N, clip=True)

    gs = gridspec.GridSpec(1, 2, width_ratios=[0.9, 0.05], wspace=0.3)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(gs[0, 0])

    if is_generated:
        ax.scatter(data[:, 0], data[:, 1], c=values, cmap=cmap, norm=norm, s=25, edgecolor='k')
    else:
        ax.scatter(data[:, 0], data[:, 1], c=values, cmap=cmap, norm=norm, s=25)
    
    if is_generated:
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])

    ax.set_xlabel(' ')
    ax.set_ylabel(' ')
    ax.set_title(' ')

    cax = fig.add_subplot(gs[0, 1])
    plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=label)
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close(fig)


# -----------------------------
# Dataset / model utils
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss_weighting_type", type=str, default="constant", choices=["constant", "min_snr"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--nearest_k", type=int, default=5)
    return parser.parse_args()


def load_dataset(dataset_name, num_samples, batch_size, random_state):
    dataset_classes = {
        "Central_Banana": CentralBananaDataset,
        "Moon_with_scatterings": MoonWithScatteringsDataset,
        "Moon_with_two_circles_bounded": MoonWithTwoCiclesBoundedDataset,
        "Moon_with_two_circles_unbounded": MoonWithTwoCirclesUnboundedDataset,
        "Swiss_Roll": SwissRollDataset,
    }
    ds = dataset_classes[dataset_name](num_samples, random_state)
    X = ds.generate()
    X_train, _ = train_test_split(X, test_size=0.2, random_state=random_state)
    X_output = torch.tensor(X_train, dtype=torch.float32)
    dl = DataLoader(
        TensorDataset(X_output),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    return ds, dl, X_output, X


def create_model(reg, schedule_type, learning_rate):
    eps_model = ConditionalDenseModel([2, 128, 128, 128, 2], activation="relu", embed_dim=128)
    diffusion_steps = 200 
    betas = make_beta_schedule(num_steps=diffusion_steps, mode=schedule_type, beta_range=(1e-04, 0.02))
    return ddpm(eps_model=eps_model, betas=betas, criterion="mse", lr=learning_rate, reg=reg)


def train(model, train_loader, device, loss_weighting_type, steps, dataset_name):
    model.to(device).train()
    step = 0
    data_iter = iter(train_loader)
    pbar = tqdm(total=steps, desc="Training")

    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        loss, _, _ = model.train_step(batch[0].to(device, non_blocking=True), loss_weighting_type)
        step += 1
        pbar.update(1)
    pbar.close()
    return model


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    reg_values = [0.0]
    datasets = [
        "Central_Banana",
        "Moon_with_scatterings",
        "Moon_with_two_circles_bounded",
        "Moon_with_two_circles_unbounded",
        "Swiss_Roll",
    ]

    main_log_dir = f"logs/density_plots_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(main_log_dir, exist_ok=True)

    for dataset_name in datasets:
        dataset_log_dir = os.path.join(main_log_dir, dataset_name)
        os.makedirs(dataset_log_dir, exist_ok=True)

        ds, train_loader, X_train, X_full = load_dataset(dataset_name, args.num_samples, args.batch_size, args.seed)
        X_tensor = torch.tensor(X_full, dtype=torch.float32).to(device)

        # Training data density (radii on real vs real; leave-one-out)
        radii_train = knn_radii(X_full, X_full, args.nearest_k, leave_one_out=True)
        plot_color_coded_points(X_full, radii_train,
                                save_path=os.path.join(dataset_log_dir, f"{dataset_name}_training_density.png"),
                                is_generated=False)

        # Model training and generated data density plots
        for reg in reg_values:
            reg_log_dir = os.path.join(dataset_log_dir, f"reg_{reg}")
            os.makedirs(reg_log_dir, exist_ok=True)

            for run_idx in range(args.runs):
                print(f"\n--- Dataset: {dataset_name} | Run {run_idx+1}/{args.runs} | reg={reg} ---")
                model = create_model(reg, args.schedule, args.lr)
                model = train(model, train_loader, device, args.loss_weighting_type, args.steps, dataset_name)
                model.eval()

                x_gen = model.generate(sample_shape=X_tensor[0].shape, num_samples=args.num_samples)[0].cpu().numpy()

                # Counts and radii for generated points w.r.t. real
                counts_gen, _ = count_points_within_radius(X_full, x_gen, args.nearest_k)
                plot_color_coded_points(x_gen, counts_gen,
                                        save_path=os.path.join(reg_log_dir, f"{dataset_name}_generated_run_{run_idx+1}.png"),
                                        is_generated=True)