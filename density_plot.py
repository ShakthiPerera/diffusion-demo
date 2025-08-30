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
    lo, hi = float(np.min(radii)), float(np.max(radii))
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    edges = [
        lo,
        lo + 0.015 * (hi - lo),
        lo + 0.03 * (hi - lo),
        lo + 0.2 * (hi - lo),
        hi
    ]
    edges = np.maximum.accumulate(edges)
    if len(np.unique(edges)) < 5:
        edges = np.linspace(lo, hi, 5)
    return np.array(edges, dtype=float)


def plot_radii_bins(data: np.ndarray, radii: np.ndarray, save_path: str, bins: np.ndarray = None):
    if bins is None:
        bins = get_asymmetric_bins_radii(radii)
    colors = ['red', 'lightcoral', 'lightblue', 'blue']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bins, cmap.N, clip=True)

    gs = gridspec.GridSpec(1, 2, width_ratios=[0.9, 0.05], wspace=0.3)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(gs[0, 0])

    ax.scatter(data[:, 0], data[:, 1], c=radii, cmap=cmap, norm=norm, s=25, edgecolor='k')
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')
    ax.set_title(' ')

    cax = fig.add_subplot(gs[0, 1])
    plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=' ')
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    parser.add_argument("--lr", type=float, default=1e-2)
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
        "GMM": GMMDataset
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


def create_model(dataset_name, reg, schedule_type, learning_rate):
    eps_model = ConditionalDenseModel([2, 128, 128, 128, 2], activation="relu", embed_dim=128)
    diffusion_steps = 200 if dataset_name != "GMM" else 1000
    betas = make_beta_schedule(num_steps=diffusion_steps, mode=schedule_type, beta_range=(1e-04, 0.02))
    return ddpm(eps_model=eps_model, betas=betas, criterion="mse", lr=learning_rate, reg=reg)


def train(model, train_loader, device, loss_weighting_type, steps, dataset_name):
    model.to(device).train()
    step_size = 20000 if dataset_name == "GMM" else 5000
    scheduler = StepLR(model.optimizer, step_size=step_size, gamma=0.1)

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
        scheduler.step()
        pbar.update(1)
    pbar.close()
    return model


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    reg_values = [0.0, 0.3, 0.5, 0.8, 1.0]
    datasets = [
        "Central_Banana",
        "Moon_with_scatterings",
        "Moon_with_two_circles_bounded",
        "Moon_with_two_circles_unbounded",
        "Swiss_Roll",
        "GMM"
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
        bins_train = get_asymmetric_bins_radii(radii_train)
        plot_radii_bins(X_full, radii_train,
                        save_path=os.path.join(dataset_log_dir, f"{dataset_name}_training_density.png"),
                        bins=bins_train)

        # Longer training for GMM
        if dataset_name == "GMM":
            args.steps = 100000

        # Model training and generated data density plots
        for reg in reg_values:
            reg_log_dir = os.path.join(dataset_log_dir, f"reg_{reg}")
            os.makedirs(reg_log_dir, exist_ok=True)

            for run_idx in range(args.runs):
                print(f"\n--- Dataset: {dataset_name} | Run {run_idx+1}/{args.runs} | reg={reg} ---")
                model = create_model(dataset_name, reg, args.schedule, args.lr)
                model = train(model, train_loader, device, args.loss_weighting_type, args.steps, dataset_name)
                model.eval()

                x_gen = model.generate(sample_shape=X_tensor[0].shape, num_samples=args.num_samples)[0].cpu().numpy()

                # Radii for generated points w.r.t. real (use same bins)
                radii_gen = knn_radii(X_full, x_gen, args.nearest_k, leave_one_out=False)
                plot_radii_bins(x_gen, radii_gen,
                                save_path=os.path.join(reg_log_dir, f"{dataset_name}_generated_run_{run_idx+1}.png"),
                                bins=bins_train)
