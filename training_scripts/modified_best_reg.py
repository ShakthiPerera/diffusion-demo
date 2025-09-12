import argparse
import os
import csv
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from prdc import compute_prdc
import sys

project_path = 'diffusion-demo'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from datasets import (
    CentralBananaDataset, MoonWithScatteringsDataset,
    MoonWithTwoCirclesUnboundedDataset, SwissRollDataset
)
from diffusion import ConditionalDenseModel
from functions import make_beta_schedule
from diffusion.ddpm_ema import DDPM as ddpm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--schedule", type=str, default="cosine")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--lr", type=float, default=0.0002)  # Conservative LR for stability
    parser.add_argument("--loss_weighting_type", type=str, default="constant", choices=["constant", "min_snr"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="Moon_with_two_circles_unbounded", 
                       choices=["Central_Banana", "Moon_with_scatterings", 
                                "Moon_with_two_circles_unbounded", "Swiss_Roll"])
    return parser.parse_args()

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dataset(dataset_name, num_samples, batch_size, random_state):
    dataset_classes = {
        "Central_Banana": CentralBananaDataset,
        "Moon_with_scatterings": MoonWithScatteringsDataset,
        "Moon_with_two_circles_unbounded": MoonWithTwoCirclesUnboundedDataset,
        "Swiss_Roll": SwissRollDataset,
    }
    ds = dataset_classes[dataset_name](2*num_samples, random_state)
    X = ds.generate()
    X_train, X_val = train_test_split(X, test_size=0.5, random_state=random_state)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, drop_last=True, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor), batch_size=batch_size, drop_last=True, shuffle=False, num_workers=2, pin_memory=True)
    return ds, train_loader, val_loader, X_train_tensor, X

def create_model(reg, schedule_type, learning_rate):
    eps_model = ConditionalDenseModel([2, 128, 128, 128, 2], activation="relu", embed_dim=12)
    betas = make_beta_schedule(num_steps=1000, mode=schedule_type, beta_range=(1e-04, 0.02))
    return ddpm(eps_model=eps_model, betas=betas, criterion="mse", lr=learning_rate, reg=reg)

def train(model, train_loader, val_loader, device, loss_weighting_type, steps):
    model.to(device).train()
    scheduler = ReduceLROnPlateau(model.optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-5, verbose=True, threshold=1e-3)
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
        if step % 1000 == 0:
            val_loss = model.validate(val_loader)
            pbar.set_postfix({"Avg Loss": f"{avg_loss:.6f}", "Val Loss": f"{val_loss:.6f}"})
            scheduler.step(val_loss)  # Step scheduler with validation loss
        pbar.update(1)
    pbar.close()
    return model

def plot_real_generated_data(x_gen, X_test, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.scatter(
        X_test[:, 0].cpu().numpy(), X_test[:, 1].cpu().numpy(),
        s=8, edgecolors='none', alpha=0.8, color="#1f77b4", label='Real'
    )
    ax.scatter(
        x_gen[:, 0].cpu().numpy(), x_gen[:, 1].cpu().numpy(),
        s=8, edgecolors='none', alpha=0.7, color="#ff7f0e", label='Generated'
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
    reg_values = np.linspace(0.2, 0.4, 10)
    datasets = [args.dataset]
    main_log_dir = f"{args.logdir}/{args.dataset}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(main_log_dir, exist_ok=True)

    with open(os.path.join(main_log_dir, "settings.txt"), "w") as f:
        f.write("Training Settings\n=================\n\n")
        for arg, val in vars(args).items():
            f.write(f"{arg}: {val}\n")
        f.write(f"Regularization values: {list(np.round(reg_values, 4))}\n")
        f.write("\nModel Settings\nHidden layers: [2, 128, 128, 128, 2]\nActivation: relu\nEmbedding dim: 2\nBeta schedule: num_steps=1000, range=(1e-4, 0.02)\nEMA decay: 0.995\nOptimizer: AdamW with weight_decay=0.01\n")

    for dataset_name in datasets:
        dataset_log_dir = os.path.join(main_log_dir, dataset_name)
        os.makedirs(dataset_log_dir, exist_ok=True)
        ds, train_loader, val_loader, X_train, X = load_dataset(dataset_name, args.num_samples, args.batch_size, args.seed)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        combined_metrics = []
        for reg in reg_values:
            reg_log_dir = os.path.join(dataset_log_dir, f"reg_{reg:.4f}")
            os.makedirs(reg_log_dir, exist_ok=True)
            metrics_runs = []

            for run_idx in range(args.runs):
                print(f"\n--- Dataset: {dataset_name} | Run {run_idx+1}/{args.runs} | reg={reg:.4f} ---")
                set_seed(args.seed + run_idx)
                model = create_model(reg, args.schedule, args.lr)
                model = train(model, train_loader, val_loader, device, args.loss_weighting_type, args.steps)
                model.eval()
                with torch.no_grad():
                    x_gen = model.generate(sample_shape=X_tensor[0].shape, num_samples=args.num_samples)
                prdc_ = compute_prdc(real_features=X_tensor.cpu().numpy(), fake_features=x_gen[0].cpu().numpy(), nearest_k=5)
                metrics_runs.append(prdc_)
                plot_real_generated_data(x_gen[0], X_tensor, os.path.join(reg_log_dir, f'generated_vs_real_run_{run_idx+1}.png'))

            metrics_keys = ['precision', 'recall', 'density', 'coverage']
            metrics_mean_std = {'Reg': f"{reg:.4f}"}
            means = {}
            for k in metrics_keys:
                values = [r[k] for r in metrics_runs]
                mean_val = np.mean(values)
                std_val = np.std(values)
                metrics_mean_std[k] = f"{mean_val:.4f} ± {std_val:.4f}"
                means[k] = mean_val
            combined_metrics.append({'Reg': reg, 'mean_std': metrics_mean_std, 'means': means})

        reg0_means = next((entry['means'] for entry in combined_metrics if abs(entry['Reg']) < 1e-6), None)
        if reg0_means:
            for entry in combined_metrics:
                for k in metrics_keys:
                    mean_val = entry['means'][k]
                    reg0_mean = reg0_means[k]
                    abs_diff = mean_val - reg0_mean
                    pct_diff = ((mean_val - reg0_mean) / reg0_mean * 100) if reg0_mean != 0 else 0.0
                    entry['mean_std'][f"{k}_abs_diff"] = f"{abs_diff:.4f}"
                    entry['mean_std'][f"{k}_pct_diff"] = f"{pct_diff:.2f}%"

        csv_path = os.path.join(dataset_log_dir, "metrics_all_regs_with_diffs.csv")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            headers = ["Reg"] + [k for k in metrics_keys] + [f"{k}_abs_diff" for k in metrics_keys] + [f"{k}_pct_diff" for k in metrics_keys]
            writer.writerow(headers)
            for entry in combined_metrics:
                row = [f"{entry['Reg']:.4f}"] + [entry['mean_std'].get(col, '0.0000 ± 0.0000') for col in metrics_keys] + \
                      [entry['mean_std'].get(f"{col}_abs_diff", '0.0000') for col in metrics_keys] + \
                      [entry['mean_std'].get(f"{col}_pct_diff", '0.00%') for col in metrics_keys]
                writer.writerow(row)
        print(f"\nMetrics for {dataset_name} saved in: {csv_path}")