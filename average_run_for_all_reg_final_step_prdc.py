import argparse
import os
import csv
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from prdc import compute_prdc
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

from datasets import (
    CentralBananaDataset, MoonWithScatteringsDataset, MoonWithTwoCiclesBoundedDataset,
    MoonWithTwoCirclesUnboundedDataset, MoonDataset, SCurveDataset, SwissRollDataset, GMMDataset
)
from diffusion import ConditionalDenseModel
from functions import make_beta_schedule
from diffusion.ddpm_norms import DDPM as ddpm

def compute_wasserstein_2(real, fake):
    return np.mean([wasserstein_distance(real[:, i], fake[:, i]) for i in range(real.shape[1])])

def compute_mmd(X, Y, kernel_type='rbf'):
    kernel = {
        'linear': linear_kernel,
        'poly': lambda X, Y: polynomial_kernel(X, Y, degree=3),
        'rbf': lambda X, Y: rbf_kernel(X, Y, gamma=1.0 / X.shape[1])
    }[kernel_type]
    Kxx, Kyy, Kxy = kernel(X, X), kernel(Y, Y), kernel(X, Y)
    m, n = X.shape[0], Y.shape[0]
    return Kxx.sum() / (m*m) + Kyy.sum() / (n*n) - 2 * Kxy.sum() / (m*n)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--loss_weighting_type", type=str, default="constant", choices=["constant", "min_snr"])
    parser.add_argument("--runs", type=int, default=10)
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
    return ds, DataLoader(TensorDataset(X_output), batch_size=batch_size, drop_last=True, shuffle=True, num_workers=2, pin_memory=True), X_output, X

def create_model(reg, schedule_type, learning_rate):
    eps_model = ConditionalDenseModel([2, 128, 128, 128, 2], activation="relu", embed_dim=128)
    betas = make_beta_schedule(num_steps=1000, mode=schedule_type, beta_range=(1e-04, 0.02))
    return ddpm(eps_model=eps_model, betas=betas, criterion="mse", lr=learning_rate, reg=reg)

def train(model, train_loader, device, loss_weighting_type, steps):
    model.to(device).train()
    # scheduler = StepLR(model.optimizer, step_size=20000, gamma=0.1)
    step = 0
    data_iter = iter(train_loader)
    pbar = tqdm(total=steps, desc="Training")
    total_loss = 0.0
    
    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        loss, _, _ = model.train_step(batch[0].to(device, non_blocking=True), loss_weighting_type)
        total_loss += loss
        step += 1
        avg_loss = total_loss / step
        if step % 1000 == 1:
            pbar.set_postfix({"Avg Loss": f"{avg_loss:.6f}", "LR": f"{model.optimizer.param_groups[0]['lr']:.6f}"})
        # scheduler.step()
        pbar.update(1)
    pbar.close()
    return model

def plot_real_generated_data(x_gen, X_test, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.scatter(
    X_test[:, 0].cpu().numpy(), X_test[:, 1].cpu().numpy(),
    s=10, edgecolors='none', alpha=0.4, color="#1f77b4", label='Real'   # Blue
)
    ax.scatter(
        x_gen[:, 0].cpu().numpy(), x_gen[:, 1].cpu().numpy(),
    s=10, edgecolors='none', alpha=0.6, color="#ff7f0e", label='Generated'  # Orange
)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', color='gray', alpha=0.2, linestyle='-', zorder=0)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=10, loc='best')
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    args = parse_args()
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    reg_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # datasets = ["Central_Banana", "Moon_with_scatterings", "Moon_with_two_circles_bounded", 
                # "Moon_with_two_circles_unbounded", "Swiss_Roll", "GMM"]
    datasets = ["GMM"]
    main_log_dir = f"logs/final_step_running_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(main_log_dir, exist_ok=True)

    with open(os.path.join(main_log_dir, "settings.txt"), "w") as f:
        f.write("Training Settings\n=================\n\n")
        for arg, val in vars(args).items():
            f.write(f"{arg}: {val}\n")
        f.write("\nModel Settings\nHidden layers: [2, 128, 128, 128, 2]\nActivation: relu\nEmbedding dim: 128\nBeta schedule: num_steps=1000, range=(1e-4, 0.02)\n")

    for dataset_name in datasets:
        dataset_log_dir = os.path.join(main_log_dir, dataset_name)
        os.makedirs(dataset_log_dir, exist_ok=True)
        ds, train_loader, X_train, X = load_dataset(dataset_name, args.num_samples, args.batch_size, args.seed)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        combined_metrics = []
        for reg in reg_values:
            reg_log_dir = os.path.join(dataset_log_dir, f"reg_{reg}")
            os.makedirs(reg_log_dir, exist_ok=True)
            # metrics_runs = []

            for run_idx in range(args.runs):
                print(f"\n--- Dataset: {dataset_name} | Run {run_idx+1}/{args.runs} | reg={reg} ---")
                model = create_model(reg, args.schedule, args.lr)
                model = train(model, train_loader, device, args.loss_weighting_type, args.steps)
                model.eval()
                x_gen = model.generate(sample_shape=X_tensor[0].shape, num_samples=args.num_samples)
                # prdc_ = compute_prdc(real_features=X_tensor.cpu().numpy(), fake_features=x_gen[0].cpu().numpy(), nearest_k=5)
                # metrics_runs.append({
                #     **prdc_, 
                #     'wasserstein': compute_wasserstein_2(X_tensor.cpu().numpy(), x_gen[0].cpu().numpy()),
                #     'linear_mmd': compute_mmd(X_tensor.cpu().numpy(), x_gen[0].cpu().numpy(), 'linear'),
                #     'poly_mmd': compute_mmd(X_tensor.cpu().numpy(), x_gen[0].cpu().numpy(), 'poly'),
                #     'rbf_mmd': compute_mmd(X_tensor.cpu().numpy(), x_gen[0].cpu().numpy(), 'rbf')
                # })
                plot_real_generated_data(x_gen[0], X_tensor, os.path.join(reg_log_dir, f'generated_vs_real_run_{run_idx+1}.png'))

        #     metrics_mean_std = {'Reg': reg}
        #     for k in metrics_runs[0].keys():
        #         values = [r[k] for r in metrics_runs]
        #         metrics_mean_std[k] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
        #     combined_metrics.append(metrics_mean_std)

        # csv_path = os.path.join(dataset_log_dir, "metrics_all_regs.csv")
        # with open(csv_path, mode='w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Reg", "precision", "recall", "density", "coverage", "wasserstein", "linear_mmd", "poly_mmd", "rbf_mmd"])
        #     for stats in combined_metrics:
        #         writer.writerow([stats.get(col, '') for col in ["Reg", "precision", "recall", "density", "coverage", "wasserstein", "linear_mmd", "poly_mmd", "rbf_mmd"]])
        # print(f"\nMetrics for {dataset_name} saved in: {csv_path}")