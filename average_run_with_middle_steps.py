import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from datetime import datetime
from collections import defaultdict

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
    parser.add_argument("--steps", type=positive_int, default=40000)
    parser.add_argument("--val_step", type=positive_int, default=1000)
    parser.add_argument("--seed", type=positive_int, default=42)
    parser.add_argument("--gpu_id", type=non_negative_int, default=0)
    parser.add_argument("--num_samples", type=positive_int, default=10000)
    parser.add_argument("-b", "--batch_size", type=positive_int, default=512)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--loss_weighting_type", type=valid_loss_weighting_type,
                        default="constant", help='Loss weighting type: "constant" or "min_snr"')
    parser.add_argument("--runs", type=positive_int, default=10, help="Number of runs per reg")
    parser.add_argument("--lr_step", type=positive_int, default=10000, help="Step interval for LR decay")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="LR decay factor")
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
    X_train, _ = train_test_split(X, test_size=0.2, random_state=random_state)
    X_output = torch.tensor(X_train, dtype=torch.float32)
    train_set = TensorDataset(X_output)
    data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    return ds, data_loader, X_output, X


def create_model(reg, schedule_type, learning_rate):
    num_features = [2, 128, 128, 128, 2]
    eps_model = ConditionalDenseModel(num_features, activation="relu", embed_dim=10)
    betas = make_beta_schedule(num_steps=200, mode=schedule_type, beta_range=(1e-04, 0.02))
    model = ddpm(eps_model=eps_model, betas=betas, criterion="mse", lr=learning_rate, reg=reg)
    return model


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
    validation_steps = range(0, args.steps + 1, args.val_step)

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    main_log_dir = f"logs/{args.dataset}_{date_str}_{time_str}"
    os.makedirs(main_log_dir, exist_ok=True)

    combined_prdc_stats = []

    ds, train_loader, X_train, X = load_dataset(args.dataset, args.num_samples, args.batch_size, args.seed)
    X_tensor = X_train.to(device)

    for reg in reg_values:
        reg_log_dir = os.path.join(main_log_dir, f"reg_{reg}")
        os.makedirs(reg_log_dir, exist_ok=True)

        prdc_list = []

        for run_idx in range(args.runs):
            print(f"\n--- Run {run_idx+1}/{args.runs} | reg={reg} ---")
            model = create_model(reg, args.schedule, args.lr)
            model.to(device)
            model.train()

            # --- Optimizer and LR scheduler ---
            optimizer = model.optimizer
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.lr_step, gamma=args.lr_gamma
            )

            step = 0
            data_iter = iter(train_loader)
            pbar = tqdm(total=args.steps, desc=f"Training reg={reg}", dynamic_ncols=True)

            while step < args.steps + 1:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                x_batch = batch[0].to(device, non_blocking=True)
                loss, simple_loss, norm_loss = model.train_step(x_batch, args.loss_weighting_type)

                # Step the LR scheduler
                scheduler.step()

                if step in validation_steps:
                    model.eval()
                    x_gen = model.generate(sample_shape=X_tensor[0].shape, num_samples=10000)
                    prdc_ = compute_prdc(
                        real_features=X_tensor.cpu().numpy(),
                        fake_features=x_gen[0].cpu().numpy(),
                        nearest_k=5
                    )
                    prdc_list.append((step, prdc_))

                    # Save plot
                    plot_path = os.path.join(
                        reg_log_dir, f'generated_vs_real_run_{run_idx+1}_step_{step}.png'
                    )
                    plot_real_generated_data(x_gen[0], X_tensor, save_path=plot_path)
                    model.train()

                step += 1
                pbar.update(1)
            pbar.close()

        # Aggregate mean ± std for each step
        prdc_by_step = defaultdict(list)
        for step, prdc_dict in prdc_list:
            prdc_by_step[step].append(prdc_dict)

        for step, prdc_dicts in prdc_by_step.items():
            mean_std_dict = {"reg": reg, "step": step}
            for k in prdc_dicts[0].keys():
                values = [d[k] for d in prdc_dicts]
                mean_std_dict[k] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
            combined_prdc_stats.append(mean_std_dict)

