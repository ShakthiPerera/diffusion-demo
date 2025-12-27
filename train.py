"""Simple training script for DDPM and ISO-DDPM on 2D toy datasets."""

from __future__ import annotations

import argparse
import os
import random
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.datasets import Synthetic2DDataset
from src.ddpm import DiffusionModel, TimeConditionedMLP
from src.metrics import compute_prdc
from src.schedules import make_beta_schedule


DATASETS = ("moon_scatter", "swiss_roll", "central_banana", "moon_circles")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a compact DDPM/ISO-DDPM on 2D datasets.")
    parser.add_argument("--dataset", type=str, choices=DATASETS, default="moon_scatter")
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--num_diffusion_steps", type=int, default=1000)
    parser.add_argument("--schedule", type=str, choices=["cosine", "linear", "quadratic"], default="cosine")
    parser.add_argument("--train_steps", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--reg_strength", type=float, default=0.0, help="ISO regularisation strength.")
    parser.add_argument("--nearest_k", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--sample_size", type=int, default=10_000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloader(data: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    tensor = torch.tensor(data, dtype=torch.float32)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True, drop_last=True, generator=gen)


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.random_state)

    dataset = Synthetic2DDataset(
        name=args.dataset,
        num_samples=args.num_samples,
        noise_level=args.noise_level,
        random_state=args.random_state,
    )
    data = dataset.generate()
    dataloader = make_dataloader(data, args.batch_size, args.random_state)
    run_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, f"true_{args.dataset}.npy"), data.astype(np.float32))

    betas = make_beta_schedule(args.num_diffusion_steps, mode=args.schedule)
    eps_model = TimeConditionedMLP(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_steps=args.num_diffusion_steps,
        embed_dim=args.embed_dim,
    )
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = DiffusionModel(
        eps_model=eps_model,
        betas=betas,
        lr=args.lr,
        ema_decay=args.ema_decay,
        device=device,
        reg_strength=args.reg_strength,
    ).to(device)

    # write a simple run sheet with model/diffusion settings
    settings_path = os.path.join(run_dir, "settings.txt")
    with open(settings_path, "w") as sf:
        sf.write("Dataset\n")
        sf.write(f"  name: {args.dataset}\n")
        sf.write(f"  num_samples: {args.num_samples}\n")
        sf.write(f"  noise_level: {args.noise_level}\n")
        sf.write(f"  random_state: {args.random_state}\n\n")
        sf.write("Diffusion\n")
        sf.write(f"  num_diffusion_steps: {args.num_diffusion_steps}\n")
        sf.write(f"  schedule: {args.schedule}\n")
        sf.write(f"  reg_strength: {args.reg_strength}\n")
        sf.write(f"  ema_decay: {args.ema_decay}\n\n")
        sf.write("Model\n")
        sf.write(f"  hidden_dim: {args.hidden_dim}\n")
        sf.write(f"  num_layers: {args.num_layers}\n")
        sf.write(f"  embed_dim: {args.embed_dim}\n")
        sf.write("\nTraining\n")
        sf.write(f"  train_steps: {args.train_steps}\n")
        sf.write(f"  batch_size: {args.batch_size}\n")
        sf.write(f"  lr: {args.lr}\n")
        sf.write(f"  log_every: {args.log_every}\n")
        sf.write("\nSampling\n")
        sf.write(f"  sample_size: {args.sample_size}\n")
        sf.write(f"  nearest_k: {args.nearest_k}\n")

    data_iter = iter(dataloader)
    for step in range(args.train_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        x0 = batch[0].to(device)
        loss_val = model.train_step(x0)
        if (step + 1) % args.log_every == 0 or step == 0:
            print(f"Step {step + 1}/{args.train_steps} - loss: {loss_val:.6f}")

    print("Training complete. Sampling...")
    samples = model.sample(args.sample_size, device=device, use_ema=args.ema_decay is not None)
    samples_np = samples.cpu().numpy().astype(np.float32)
    metrics = compute_prdc(real_features=data, fake_features=samples_np, nearest_k=args.nearest_k)

    np.save(os.path.join(run_dir, f"generated_{args.dataset}.npy"), samples_np)
    with open(os.path.join(run_dir, f"prdc_{args.dataset}.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(data[:, 0], data[:, 1], s=2, alpha=0.5, label="train")
        ax.scatter(samples_np[:, 0], samples_np[:, 1], s=2, alpha=0.5, label="generated")
        ax.set_title(f"{args.dataset} - reg={args.reg_strength}")
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"plot_{args.dataset}.png"), dpi=300)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting not critical
        print(f"Plotting failed: {exc}")

    print("PRDC:", metrics)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train(args)
