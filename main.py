import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
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
    def restricted_float(x):
        x = float(x)
        if x <= 0 or x > 1:
            raise argparse.ArgumentTypeError(f"reg must be in (0, 1], got {x}")
        return x

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
    parser.add_argument(
        "--reg", type=restricted_float, default=0.0,
        help="Regularization strength in (0, 1] (default: 0.0)",
    )
    parser.add_argument(
        "--steps", type=positive_int, default=20000,
        help="Total number of steps for training.",
    )
    parser.add_argument(
        "--gpu_id", type=non_negative_int, default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--num_samples", type=positive_int, default=10000,
        help="Number of samples for dataset creation",
    )
    parser.add_argument(
        "--seed", type=positive_int, default=42,
        help="Random state for reproducibility",
    )
    parser.add_argument(
        "-b", "--batch_size", type=positive_int, default=512,
        help="Batch size for training",
    )
    parser.add_argument("--schedule", type=str, default="linear", help="Beta schedule type")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--lr_step_size", type=int, default=10000, help="Step size for StepLR")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="Gamma factor for StepLR")

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


def train(model, train_loader, device, loss_weighting_type, steps=150000, lr_scheduler=None):
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

        if lr_scheduler is not None:
            lr_scheduler.step()  # update learning rate each step

        if step % 1000 == 0:  # Reduce tqdm updates for speed
            current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else model.lr
            pbar.set_postfix({
                'step': step + 1,
                'loss': train_losses[:step+1].mean(),
                'simple_loss': diff_losses[:step+1].mean(),
                'norm_loss': norm_losses[:step+1].mean(),
                'lr': current_lr
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

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    log_dir = f"logs/{args.dataset}_{date_str}_{time_str}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Loss Weighting Type: {args.loss_weighting_type}")
    print(f"Reg: {args.reg}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Num Samples: {args.num_samples}")
    print(f"Random State: {args.seed}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Log Dir: {log_dir}")

    ds, train_loader, X_train, X = load_dataset(args.dataset, args.num_samples, args.batch_size, args.seed)
    model = create_model(args.reg, args.schedule, args.lr)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    # --- StepLR scheduler added ---
    optimizer = model.optimizer  # assumes ddpm class has an attribute .optimizer
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    train_losses, diff_losses, norm_losses = train(
        model, train_loader, device,
        loss_weighting_type=args.loss_weighting_type,
        steps=args.steps,
        lr_scheduler=lr_scheduler
    )

    X_train = X_train.to(device)
    model.eval()
    X = torch.Tensor(X).to(device)
    x_gen = model.generate(sample_shape=X[0].shape, num_samples=10000)
    prdc_ = compute_prdc(real_features=X.cpu().numpy(), fake_features=x_gen[0].cpu().numpy(), nearest_k=5)
    print(f"PRDC: {prdc_}")

    plot_real_generated_data(x_gen[0], X, save_path=os.path.join(log_dir, 'plot.png'))
