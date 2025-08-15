import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from datetime import datetime

import torch
from torch.utils.data import TensorDataset, DataLoader

from datasets import (
    BananaWithTwoCirclesDataset,
    BananaDataset,
    CentralBananaDataset,
    EightGaussiansDataset,
    MoonWithScatteringsDataset,
    MoonWithTwoCiclesBoundedDataset,
    MoonWithTwoCirclesUnboundedDataset,
    MoonDataset,
    MultimodalGasussiansDataset,
    SCurveDataset,
    StarFishDecayDataset,
    StarFishUniformDataset,
    SwissRollDataset,
    TwoRingsBoundedDataset,
)
from diffusion import ConditionalDenseModel
from functions import make_beta_schedule
from diffusion.ddpm_norms import DDPM as ddpm
from prdc import compute_prdc


def parse_args():
    def positive_int(x):
        x = int(x)
        if x <= 0:
            raise argparse.ArgumentTypeError(f"{x} must be greater than 0")
        return x

    def non_negative_int(x):
        x = int(x)
        if x < 0:
            raise argparse.ArgumentTypeError(f"{x} must be non-negative")
        return x

    def valid_loss_weighting_type(x):
        x = x.lower()
        if x not in ["constant", "min_snr"]:
            raise argparse.ArgumentTypeError(
                f"loss_weighting_type must be 'constant' or 'min_snr', got {x}"
            )
        return x

    parser = argparse.ArgumentParser(description="Model training script")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--loss_weighting_type",
        type=valid_loss_weighting_type,
        default="constant",
        help='Loss weighting type: "constant" or "min_snr"',
    )
    parser.add_argument("--steps", type=positive_int, default=20000)
    parser.add_argument("--gpu_id", type=non_negative_int, default=0)
    parser.add_argument("--num_samples", type=positive_int, default=10000)
    parser.add_argument("--seed", type=positive_int, default=42)
    parser.add_argument("-b", "--batch_size", type=positive_int, default=512)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--lr", type=float, default=1e-2)

    return parser.parse_args()


def load_dataset(dataset_name, num_samples, batch_size, random_state):
    dataset_classes = {
        "Banana_with_two_circles": BananaWithTwoCirclesDataset,
        "Banana": BananaDataset,
        "Central_Banana": CentralBananaDataset,
        "8_Gaussians": EightGaussiansDataset,
        "Moon_with_scatterings": MoonWithScatteringsDataset,
        "Moon_with_two_circles_bounded": MoonWithTwoCiclesBoundedDataset,
        "MoonWithTwoCirclesUnboundedDataset": MoonWithTwoCirclesUnboundedDataset,
        "Moons": MoonDataset,
        "Multimodal_Gaussians": MultimodalGasussiansDataset,
        "S_Curve": SCurveDataset,
        "Star_fish_decay": StarFishDecayDataset,
        "Star_fish_uniform": StarFishUniformDataset,
        "Swiss_Roll": SwissRollDataset,
        "Two_rings_bounded": TwoRingsBoundedDataset,
    }

    ds = dataset_classes[dataset_name](num_samples, random_state)
    X = ds.generate()
    X_train, _ = train_test_split(X, test_size=0.2)
    X_output = torch.tensor(X_train, dtype=torch.float32)
    train_set = TensorDataset(X_output)
    data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    return ds, data_loader, X_output, X


def create_model(reg, schedule_type, learning_rate):
    num_features = [2, 128, 128, 128, 2]
    eps_model = ConditionalDenseModel(num_features, activation="relu", embed_dim=10)
    betas = make_beta_schedule(
        num_steps=200, mode=schedule_type, beta_range=(1e-04, 0.02)
    )
    return ddpm(eps_model=eps_model, betas=betas, criterion="mse", lr=learning_rate, reg=reg)


def train(model, train_loader, device, loss_weighting_type, steps=150000):
    model.to(device)
    train_losses = np.empty(steps, dtype=np.float32)
    diff_losses = np.empty(steps, dtype=np.float32)
    norm_losses = np.empty(steps, dtype=np.float32)

    model.train()
    step = 0
    data_iter = iter(train_loader)
    pbar = tqdm(total=steps, desc="Training", dynamic_ncols=True)

    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x_batch = batch[0].to(device, non_blocking=True)
        loss, simple_loss, norm_loss = model.train_step(x_batch, loss_weighting_type)

        train_losses[step] = loss
        diff_losses[step] = simple_loss
        norm_losses[step] = norm_loss

        if step % 1000 == 0:
            pbar.set_postfix({
                'step': step + 1,
                'loss': train_losses[:step+1].mean(),
                'simple_loss': diff_losses[:step+1].mean(),
                'norm_loss': norm_losses[:step+1].mean()
            })

        pbar.update(1)
        step += 1

    pbar.close()
    return train_losses, diff_losses, norm_losses


def plot_real_generated_data(x_gen, X_test, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x_gen[:, 0].cpu().numpy(), x_gen[:, 1].cpu().numpy(), s=3,
               edgecolors='none', alpha=0.7, color=plt.cm.cividis(0.0), label='Generated')
    ax.scatter(X_test[:, 0].cpu().numpy(), X_test[:, 1].cpu().numpy(), s=3,
               edgecolors='none', alpha=0.5, color=plt.cm.cividis(0.8), label='Real')
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(visible=True, which='both', color='gray', alpha=0.2, linestyle='-')
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    args = parse_args()
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    reg_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Main log folder for this dataset run
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    main_log_dir = f"logs/{args.dataset}_{date_str}_{time_str}"
    os.makedirs(main_log_dir, exist_ok=True)

    all_prdc_values = []

    for reg in reg_values:
        reg_log_dir = os.path.join(main_log_dir, f"reg_{reg}")
        os.makedirs(reg_log_dir, exist_ok=True)

        ds, train_loader, X_train, X = load_dataset(args.dataset, args.num_samples, args.batch_size, args.seed)
        X_tensor = torch.Tensor(X).to(device)

        prdc_values = []

        for run_idx in range(3):
            print(f"\n--- Run {run_idx+1}/3 | reg={reg} ---")
            model = create_model(reg, args.schedule, args.lr)

            train_losses, diff_losses, norm_losses = train(
                model, train_loader, device, loss_weighting_type=args.loss_weighting_type, steps=args.steps
            )

            model.eval()
            x_gen = model.generate(sample_shape=X_tensor[0].shape, num_samples=10000)
            prdc_ = compute_prdc(real_features=X_tensor.cpu().numpy(),
                                 fake_features=x_gen[0].cpu().numpy(),
                                 nearest_k=5)
            prdc_['Reg'] = reg  # Add reg info
            prdc_values.append(prdc_)
            all_prdc_values.append(prdc_)  # Save for combined CSV

            plot_path = os.path.join(reg_log_dir, f'generated_vs_real_plot_run_{run_idx+1}.png')
            plot_real_generated_data(x_gen[0], X_tensor, save_path=plot_path)

        # Save TXT info inside reg folder
        txt_path = os.path.join(reg_log_dir, "run_info.txt")
        with open(txt_path, "w") as f:
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Loss Weighting Type: {args.loss_weighting_type}\n")
            f.write(f"Reg: {reg}\n")
            f.write(f"Steps: {args.steps}\n")
            f.write(f"GPU ID: {args.gpu_id}\n")
            f.write(f"Num Samples: {args.num_samples}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Learning Rate: {args.lr}\n")
            f.write(f"Schedule: {args.schedule}\n")
            f.write(f"Log Dir: {reg_log_dir}\n")
            f.write(f"Run Date: {date_str} {time_str}\n")
            f.write("\nPRDC Metrics (Mean ± Std):\n")
            keys = list(prdc_values[0].keys())
            for k in keys:
                mean_val = np.mean([p[k] for p in prdc_values])
                std_val = np.std([p[k] for p in prdc_values])
                f.write(f"{k}: {mean_val:.4f} ± {std_val:.4f}\n")

    # Save combined PRDC CSV outside reg folders
    csv_path = os.path.join(main_log_dir, "prdc_all_regs.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        keys = list(all_prdc_values[0].keys())
        header = ["Run"] + keys
        writer.writerow(header)
        for i, prdc_dict in enumerate(all_prdc_values, start=1):
            writer.writerow([i] + [prdc_dict[k] for k in keys])
        writer.writerow(["Mean"] + [np.mean([p[k] for p in all_prdc_values]) for k in keys])
        writer.writerow(["Std"] + [np.std([p[k] for p in all_prdc_values]) for k in keys])

    print(f"\nAll combined PRDC results saved in: {csv_path}")