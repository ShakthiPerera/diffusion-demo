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

from datasets import (
    CentralBananaDataset, MoonWithScatteringsDataset, MoonWithTwoCiclesBoundedDataset, SwissRollDataset, GMMDataset
)
from diffusion import ConditionalDenseModel
from functions import make_beta_schedule
from diffusion.ddpm_log import DDPM as ddpm


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
    parser.add_argument("--runs", type=int, default=10)
    return parser.parse_args()


def load_dataset(dataset_name, num_samples, batch_size, random_state):
    dataset_classes = {
        "Central_Banana": CentralBananaDataset,
        "Moon_with_scatterings": MoonWithScatteringsDataset,
        "Moon_with_two_circles_bounded": MoonWithTwoCiclesBoundedDataset,
        "Swiss_Roll": SwissRollDataset,
        "GMM": GMMDataset
    }
    ds = dataset_classes[dataset_name](num_samples, random_state)
    X = ds.generate()
    X_train, _ = train_test_split(X, test_size=0.2, random_state=random_state)
    X_output = torch.tensor(X_train, dtype=torch.float32)
    return ds, DataLoader(TensorDataset(X_output), batch_size=batch_size, drop_last=True, shuffle=True), X_output


def create_model(reg, schedule_type, learning_rate):
    eps_model = ConditionalDenseModel([2, 128, 128, 128, 2], activation="relu", embed_dim=128)
    betas = make_beta_schedule(num_steps=1000, mode=schedule_type, beta_range=(1e-04, 0.02))
    return ddpm(eps_model=eps_model, betas=betas, criterion="mse", lr=learning_rate, reg=reg)


def train(model, train_loader, device, loss_weighting_type, steps, csv_path):
    model.to(device).train()
    scheduler = StepLR(model.optimizer, step_size=5000, gamma=0.1)
    step = 0
    data_iter = iter(train_loader)
    pbar = tqdm(total=steps, desc="Training")

    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        loss, simple_loss, iso_val, avg_norm = model.train_step(
            batch[0].to(device, non_blocking=True),
            loss_weighting_type,
            step=step,
            csv_path=csv_path,
            save_interval=1
        )

        step += 1
        scheduler.step()
        if step % 1000 == 1:
            pbar.set_postfix({"Loss": f"{loss:.6f}"})
        pbar.update(1)

    pbar.close()
    return model


if __name__ == "__main__":
    args = parse_args()
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    reg_values = [0.0, 0.3, 0.5, 0.8, 1.0]   # adjust as needed
    # datasets = ["Central_Banana", "Moon_with_scatterings", "Moon_with_two_circles_bounded", "Swiss_Roll", "GMM"]  # adjust as needed
    datasets = ["Swiss_Roll", "GMM"]
    
    main_log_dir = f"logs/norms_iso_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(main_log_dir, exist_ok=True)

    # Save training settings once
    with open(os.path.join(main_log_dir, "settings.txt"), "w") as f:
        f.write("Training Settings\n=================\n\n")
        for arg, val in vars(args).items():
            f.write(f"{arg}: {val}\n")
        f.write("\nModel Settings\nHidden layers: [2, 128, 128, 128, 2]\nActivation: relu\nEmbedding dim: 128\n")
        f.write("Beta schedule: num_steps=1000, range=(1e-4, 0.02)\n")

    for dataset_name in datasets:
        dataset_log_dir = os.path.join(main_log_dir, dataset_name)
        os.makedirs(dataset_log_dir, exist_ok=True)
        ds, train_loader, X_train = load_dataset(dataset_name, args.num_samples, args.batch_size, args.seed)

        for reg in reg_values:
            reg_log_dir = os.path.join(dataset_log_dir, f"reg_{reg}")
            os.makedirs(reg_log_dir, exist_ok=True)

            for run_idx in range(args.runs):
                print(f"\n--- Dataset: {dataset_name} | Run {run_idx+1}/{args.runs} | reg={reg} ---")
                model = create_model(reg, args.schedule, args.lr)
                csv_path = os.path.join(reg_log_dir, f"norms_iso_run_{run_idx+1}.csv")

                # train logs directly to CSV
                model = train(model, train_loader, device, args.loss_weighting_type, args.steps, csv_path)

                print(f"Run {run_idx+1} finished. Norms+Iso values saved to {csv_path}")
