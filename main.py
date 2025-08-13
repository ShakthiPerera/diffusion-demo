import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import GradScaler, autocast

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
from functions.stat import (
    kurtosis_,
    whiteness_measures,
    isotropy,
    metric_plot,
    kurtosis_plot
)
from prdc import compute_prdc


def parse_args():
    def restricted_float(x):
        """Ensure reg is between 0 and 1, excluding 0."""
        x = float(x)
        if x <= 0 or x > 1:
            raise argparse.ArgumentTypeError(f"reg must be in (0, 1], got {x}")
        return x

    def positive_int(x):
        """Ensure the integer is greater than 0."""
        x = int(x)
        if x <= 0:
            raise argparse.ArgumentTypeError(f"{x} must be greater than 0")
        return x

    def non_negative_int(x):
        """Ensure the integer is non-negative (for gpu_id)."""
        x = int(x)
        if x < 0:
            raise argparse.ArgumentTypeError(f"{x} must be non-negative")
        return x

    def valid_loss_weighting_type(x):
        "Restrict loss weighting to 'constant' or 'min_snr'."
        x = x.lower()
        if x not in ["constant", "min_snr"]:
            raise argparse.ArgumentTypeError(
                f"loss_weighting_type must be 'constant' or 'min_snr', got {x}"
            )
        return x
    
    def valid_date(x):
        """Validate date in YYYY-MM-DD format."""
        try:
            datetime.strptime(x, "%Y-%m-%d")
            return x
        except ValueError:
            raise argparse.ArgumentTypeError(f"Date must be in YYYY-MM-DD format, got {x}")

    def valid_time(x):
        """Validate time in HH:MM:SS format."""
        try:
            datetime.strptime(x, "%H:%M:%S")
            return x
        except ValueError:
            raise argparse.ArgumentTypeError(f"Time must be in HH:MM:SS format, got {x}")

    parser = argparse.ArgumentParser(description="Model training script")

    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (required) {}"
    )
    parser.add_argument(
        "--loss_weighting_type",
        type=valid_loss_weighting_type,
        default="constant",
        help='Loss weighting type: "constant" or "min_snr" (default: constant)',
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default= " ", 
        help="Directory to save logs (default: " "). If it is " ", a log won't be saved",
    )
    parser.add_argument(
        "--reg",
        type=restricted_float,
        default=0.0,
        help="Regularization strength in (0, 1] (default: 0.0)",
    )
    parser.add_argument(
        "--steps",
        type=positive_int,
        default = 150000,
        help="Total number of steps/number of iterations in training. (default: 150000)",
    )
    parser.add_argument(
        "--gpu_id",
        type=non_negative_int,
        default=0,
        help="GPU device ID, single non-negative integer (default: 0)",
    )
    parser.add_argument(
        "--num_samples",
        type=positive_int,
        default=10000,
        help="Number of samples for dataset creation/sampling, > 0 (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=positive_int,
        default=42,
        help="Random state number for reproducibility",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=positive_int,
        default=64,
        help="Size of the batches for training.",
    )
    parser.add_argument(
        "--schedule", type=str, default="linear", help="beta schedule type"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--hl",
        type=list,
        default=[2, 128, 128, 128, 2],
        help="hidden layers size for noise prediction",
    )
    parser.add_argument(
        "--date",
        type=valid_date,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date in YYYY-MM-DD format (default: current date)",
    )
    parser.add_argument(
        "--time",
        type=valid_time,
        default=datetime.now().strftime("%H:%M:%S"),
        help="Time in HH:MM:SS format (default: current time)",
    )
    args = parser.parse_args()
    return args


def load_dataset(dataset_name, num_samples, batch_size, random_state, purpose):
    """
    Loading the dataset for given argeparse properties

    Args:
        dataset_name (str): name of the dataset
        num_samples (int): number of samples you want
        batch_size (int) : batch size for the dataset
        random_state (int): random state fro reproducibility
        purpose (str): purpose to load the dataset ('train', 'validation')
    """
    if dataset_name == "Banana_with_two_circles":
        ds = BananaWithTwoCirclesDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "Banana":
        ds = BananaDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "Central_Banana":
        ds = CentralBananaDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "8_Gaussians":
        ds = EightGaussiansDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "Moon_with_scatterings":
        ds = MoonWithScatteringsDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "Moon_with_two_circles_bounded":
        ds = MoonWithTwoCiclesBoundedDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "MoonWithTwoCirclesUnboundedDataset":
        ds = MoonWithTwoCirclesUnboundedDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "Moons":
        ds = MoonDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "Multimodal_Gaussians":
        ds = MultimodalGasussiansDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "S_Curve":
        ds = SCurveDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "Star_fish_decay":
        ds = StarFishDecayDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "Star_fish_uniform":
        ds = StarFishUniformDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "Swiss_Roll":
        ds = SwissRollDataset(num_samples, random_state)
        X = ds.generate()
    elif dataset_name == "Two_rings_bounded":
        ds = TwoRingsBoundedDataset(num_samples, random_state)
        X = ds.generate()

    X_train, X_val = train_test_split(X, test_size=0.2)
    if purpose == "train":
        X_output = torch.tensor(X_train).float()
        train_set = TensorDataset(X_output)
        data_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=7,
            pin_memory=True,
        )
        
    elif purpose == 'validate':
        X_output = torch.tensor(X_val).float()
        val_set = TensorDataset(X_output)
        data_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=7,
            pin_memory=True,
        )
    
    return ds, data_loader , X_output


def create_model(reg, schedule_type, learning_rate, num_features):
    """
    Creating the DDPM model instance for the training

    Args:
        loss_type (str): loss type (iso, default)
        schedule_type (str): schedule variation type (linear, quadratic, cosine ..)
        learning_rate (float): models' learning rate
        num_features (list, optional): Conditional models hidden layers sizes for noise prediction. Defaults to [2, 128, 128, 128, 2].

    Returns:
        torch.model: model instanace
    """
    eps_model = ConditionalDenseModel(num_features, activation="relu", embed_dim=10)
    betas = make_beta_schedule(
        num_steps=1000, mode=schedule_type, beta_range=(1e-04, 0.02)
    )

    ddpm_model = ddpm(
        eps_model=eps_model,
        betas=betas,
        criterion="mse",
        lr=learning_rate,
        reg=reg
        )        

    return ddpm_model
'''
# Epoch based training loop
def train(model, train_loader, device, num_epochs=1000):
    model.to(device)
    scaler = GradScaler()  # For mixed precision
    train_losses = torch.zeros(num_epochs)
    diff_losses = torch.zeros(num_epochs)
    norm_losses = torch.zeros(num_epochs)
    iters = len(train_loader)
    model.train()
    for epoch in range(num_epochs):
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True
        )
        running_loss,simple_loss_,norm_loss_= 0.0, 0.0, 0.0
        for batch in pbar:
            x_batch = torch.stack(batch).to(device)
            with autocast(device_type='cuda'): # Enable mixed precision
                loss, simple_loss, norm_loss = model.train_step(x_batch)
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
            model.optimizer.zero_grad()
            # loss, simple_loss, norm_loss = model.train_step(x_batch)
            running_loss += loss.item()
            simple_loss_ += simple_loss.item()
            norm_loss_ += norm_loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))
        
        loss = running_loss/iters
        avg_diff_loss = simple_loss_ / iters
        avg_norm_loss = norm_loss_ / iters

        train_losses[epoch] = running_loss / iters
        diff_losses[epoch] = avg_diff_loss
        norm_losses[epoch] = avg_norm_loss
    
    return train_losses , diff_losses, norm_losses
'''

# Steps based training loop
def train(model, train_loader, device, loss_weighting_type, steps=150000):
    # Move model to the specified device (e.g., CPU or GPU)
    model.to(device)
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    # Initialize tensors to store losses for each step
    train_losses = torch.zeros(steps)
    diff_losses = torch.zeros(steps)
    norm_losses = torch.zeros(steps)
    # Set model to training mode
    model.train()
    
    # Initialize step counter and data iterator
    step = 0
    data_iter = iter(train_loader)
    # Initialize progress bar with total steps
    pbar = tqdm(total=steps, desc="Training", dynamic_ncols=True)
    
    # Continue training until the specified number of steps is reached
    while step < steps:
        try:
            # Get the next batch from the data loader
            batch = next(data_iter)
        except StopIteration:
            # Restart iterator if the data loader is exhausted
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        # Move batch to the specified device and stack it
        x_batch = batch[0].to(device)
        
        # Perform a single training step and get losses
        loss, simple_loss, norm_loss = model.train_step(x_batch, loss_weighting_type)
                
        # Store losses for the current step
        train_losses[step] = loss
        diff_losses[step] = simple_loss
        norm_losses[step] = norm_loss
        
        # Update progress bar with current step and loss information
        pbar.set_postfix({
            'step': step + 1,
            'loss': loss,
            'simple_loss': simple_loss,
            'norm_loss': norm_loss
        })
        # Increment progress bar and step counter
        pbar.update(1)
        step += 1
    
    # Close the progress bar
    pbar.close()
    # Return the collected losses
    return train_losses, diff_losses, norm_losses        

'''
def plot_step_by_step_noise(x_noisy,x_denoise,path):
    plot_steps = range(0, 1001, 20)
    colors = plt.cm.cividis(np.linspace(0.0, 1, len(plot_steps)))
    for time_idx, color in zip(plot_steps, colors):
        samples = x_noisy[time_idx].cpu().numpy()
        plt.scatter(samples[:,0], samples[:,1], s=4, edgecolors='none', alpha=0.5, color=color)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.title('{} steps'.format(time_idx))
        # ax.set(xticks=[], yticks=[], xlabel='', ylabel='')
        plt.tight_layout()
        plt.savefig(f"{path}/forward_diff_step_{time_idx}.png")
        plt.clf()

    # plot_steps_reverse = range(1000, -1, 20)
    colors = plt.cm.cividis(np.linspace(1.0, 0.0, len(plot_steps)))

    for time_idx, color in zip(plot_steps, colors):
        samples = x_denoise[time_idx].cpu().numpy()
        plt.scatter(samples[:,0], samples[:,1], s=4, alpha=0.7, color=color ,edgecolors='none')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.title('{} steps'.format(time_idx))
        # ax.set(xticks=[], yticks=[], xlabel='', ylabel='')
        plt.tight_layout()
        plt.savefig(f"{path}/reverse_diff_steps_part_{time_idx}.png")
        plt.clf()

def plot_real_generated_data(x_gen,X_test,path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x_gen[:,0].cpu().numpy(), x_gen[:,1].cpu().numpy(), s=3,
        edgecolors='none', alpha=0.7, color=plt.cm.cividis(0.0), label='Generated')
    ax.scatter(X_test[:,0].cpu().numpy(), X_test[:,1].cpu().numpy(), s=3,
        edgecolors='none', alpha=0.7, color=plt.cm.cividis(0.8), label='Real')
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(visible=True, which='both', color='gray', alpha=0.2, linestyle='-')
    ax.set_axisbelow(True)
    ax.legend()
    plt.savefig(f"{path}/generated_and_real_dataset.png")
    fig.tight_layout()

def plot_losses(train_losses, diff_losses, norm_losses, path):
    window_size = 30
    # Compute the moving average

    fig, ax = plt.subplots(figsize=(6, 4))
    # ax.plot(ddpm.train_losses[:], alpha=0.7, label='train')
    smoothed_train_loss = np.convolve((train_losses)[:], np.ones(window_size)/window_size, mode='same')
    ax.plot(torch.tensor(smoothed_train_loss)[:], alpha=0.7, label='smoothed_train_loss')

    # ax.plot(torch.tensor(diff_loss)[:], alpha=0.7, label='simple_diff_loss')
    smoothed_diff_loss = np.convolve((diff_losses)[:], np.ones(window_size)/window_size, mode='same')
    ax.plot(torch.tensor(smoothed_diff_loss)[:], alpha=0.7, label='smoothed_diff_loss')
    # ax.plot(ddpm.val_losses, alpha=0.7, label='val')
    ax.set(xlabel='step', ylabel='loss')
    ax.set_xlim([0, len(train_losses)])
    ax.set_ylim([0.1, 0.2])
    ax.legend()
    ax.grid(visible=True, which='both', color='gray', alpha=0.2, linestyle='-')
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.savefig(f"{path}/train_loss_and_diff_loss.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    smoothed_norm_loss = np.convolve((norm_losses.clone().detach().numpy())[:], np.ones(window_size)/window_size, mode='same')
    ax.plot(smoothed_norm_loss, alpha=0.7, label='norm_loss')
    ax.set(xlabel='step', ylabel='loss')
    ax.set_xlim([0, len(train_losses)])
    ax.set_ylim([0.0, 0.05])
    ax.legend()
    ax.grid(visible=True, which='both', color='gray', alpha=0.2, linestyle='-')
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.savefig(f"{path}/norm_loss.png")
'''

if __name__ == "__main__":
    args = parse_args()
    print(f"Dataset: {args.dataset}")
    print(f"Loss Weighting Type: {args.loss_weighting_type}")
    print(f"Log Dir: {args.log_dir}")
    print(f"Reg: {args.reg}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Num Samples: {args.num_samples}")
    print(f"Random State: {args.seed}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Hidden Layer Dims: {args.hl}")

    ds, train_loader, X_train= load_dataset(args.dataset, args.num_samples, args.batch_size, args.seed, purpose="train")
    print(type(train_loader))
    # ds.plot_dataset()
    model = create_model(args.reg, args.schedule, args.lr, args.hl)
    print(model)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    train_losses , diff_losses, norm_losses = train(model, train_loader, device, loss_weighting_type=args.loss_weighting_type, steps=args.steps)

    timestamp = f"{args.date.replace('-', '')}_{args.time.replace(':', '')}"
    main_path = f"{args.log_dir}/{timestamp}_{args.dataset}_{args.loss_weighting_type}_{args.reg}_{args.steps}_{args.num_samples}_{args.seed}_{args.batch_size}_{args.lr}"
    os.makedirs(main_path, exist_ok=True)

    # path_plots = f"{main_path}/plots"
    # os.makedirs(path_plots, exist_ok=True)

    X_train = X_train.to(device)
    x_noisy = model.diffuse_all_steps(X_train)
    x_denoise, x_eps = model.denoise_all_steps(torch.randn(10000, 2).to(device))
    # plot_step_by_step_noise(x_noisy, x_denoise, path_plots)

    model.eval()
    ds, _, X_test = load_dataset(args.dataset, args.num_samples, args.batch_size, args.seed, purpose='validate')
    X_test = X_test.to(device)
    x_gen = model.generate(sample_shape=X_test[0].shape, num_samples=10000)
    prdc_ = compute_prdc(real_features=X_test.cpu().numpy(), fake_features=x_gen.cpu().numpy(), nearest_k=5)
    print(f"PRDC: {prdc_}")

    prdc_file_path = f"{main_path}/prdc_metrics.txt"
    os.makedirs(os.path.dirname(prdc_file_path), exist_ok=True)
    with open(prdc_file_path, 'w') as f:
        f.write("PRDC Metrics:\n")
        f.write(f"Precision: {prdc_['precision']:.6f}\n")
        f.write(f"Recall: {prdc_['recall']:.6f}\n")
        f.write(f"Density: {prdc_['density']:.6f}\n")
        f.write(f"Coverage: {prdc_['coverage']:.6f}\n")

    # plot_real_generated_data(x_gen,X_test,path_plots)

    # Plotting the training losses
    # plot_losses(train_losses, diff_losses, norm_losses, path_plots)

'''
    kurtosis_values_x = np.zeros(model.num_steps)
    kurtosis_values_y = np.zeros(model.num_steps)
    kurt_norms = np.zeros(model.num_steps)
    iso_ratio = np.zeros(model.num_steps)
    frobenius_norm_difference = np.zeros(model.num_steps)
    iso_values = np.zeros(model.num_steps)

    data_after_one_step = X_train

    batch_kurtosis = kurtosis_(data_after_one_step.cpu().numpy())
    kurt_norm = np.sqrt(np.vdot(batch_kurtosis, batch_kurtosis))
    iso_r, fro_n = whiteness_measures(data_after_one_step.cpu().numpy())
    batch_isotropy = isotropy(data_after_one_step)

    kurtosis_values_x[0] = batch_kurtosis[0]
    kurtosis_values_y[0] = batch_kurtosis[1]
    kurt_norms[0] = kurt_norm
    iso_ratio[0] = iso_r
    frobenius_norm_difference[0] = fro_n
    iso_values[0] = batch_isotropy

    time_steps = model.num_steps

    for index in range(1, time_steps):
        data_after_one_step = x_noisy[index]

        batch_kurtosis = kurtosis_(data_after_one_step.cpu().numpy())
        kurt_norm = np.sqrt(np.vdot(batch_kurtosis, batch_kurtosis))
        iso_r, fro_n = whiteness_measures(data_after_one_step.cpu().numpy())
        batch_isotropy = isotropy(data_after_one_step)

        kurtosis_values_x[index] = batch_kurtosis[0]
        kurtosis_values_y[index] = batch_kurtosis[1]
        kurt_norms[index] = kurt_norm
        iso_ratio[index] = iso_r
        frobenius_norm_difference[index] = fro_n
        iso_values[index] = batch_isotropy


    kurtosis_plot(kurtosis_values_x, 'Kurtosis in x', (-3, 3), '01_forward_kurtosis_in_x',path_plots)
    kurtosis_plot(kurtosis_values_y, 'Kurtosis in y', (-3, 3), '02_forward_kurtosis_in_y',path_plots)
    kurtosis_plot(kurt_norms, 'Kurtosis Norm', (-3, 3), '03_forward_kurtosis_norm',path_plots)
    metric_plot(iso_ratio, 'Isotropy Ratio', (1.0, 2.0), '04_forward_isotropy_ratio', path_plots)
    metric_plot(frobenius_norm_difference, 'Frobenius_norm', (-0.1, 1.25), '05_forward_forbenius_norm',path_plots)
    metric_plot(iso_values, 'Isotropy Values',None ,'06_forward_isotropy_values',path_plots)

    
    kurtosis_values_x = np.zeros(model.num_steps)
    kurtosis_values_y = np.zeros(model.num_steps)
    kurt_norms = np.zeros(model.num_steps)
    iso_ratio = np.zeros(model.num_steps)
    frobenius_norm_difference = np.zeros(model.num_steps)
    iso_values = np.zeros(model.num_steps)

    data_after_one_step = x_denoise[999]

    batch_kurtosis = kurtosis_(data_after_one_step.cpu().numpy())
    kurt_norm = np.sqrt(np.vdot(batch_kurtosis, batch_kurtosis))
    iso_r, fro_n = whiteness_measures(data_after_one_step.cpu().numpy())
    batch_isotropy = isotropy(data_after_one_step)

    kurtosis_values_x[999] = batch_kurtosis[0]
    kurtosis_values_y[999] = batch_kurtosis[1]
    kurt_norms[999] = kurt_norm
    iso_ratio[999] = iso_r
    frobenius_norm_difference[999] = fro_n
    iso_values[999] = batch_isotropy

    for index in reversed(range(time_steps-1)):
        data_after_one_step = x_denoise[index]

        batch_kurtosis = kurtosis_(data_after_one_step.cpu().numpy())
        kurt_norm = np.sqrt(np.vdot(batch_kurtosis, batch_kurtosis))
        iso_r, fro_n = whiteness_measures(data_after_one_step.cpu().numpy())
        batch_isotropy = isotropy(data_after_one_step)

        kurtosis_values_x[index] = batch_kurtosis[0]
        kurtosis_values_y[index] = batch_kurtosis[1]
        kurt_norms[index] = kurt_norm
        iso_ratio[index] = iso_r
        frobenius_norm_difference[index] = fro_n
        iso_values[index] = batch_isotropy

    kurtosis_plot(kurtosis_values_x, 'Kurtosis in x', (-3, 3), '01_reverse_kurtosis_in_x',path_plots)
    kurtosis_plot(kurtosis_values_y, 'Kurtosis in y', (-3, 3), '02_reverse_kurtosis_in_y',path_plots)
    kurtosis_plot(kurt_norms, 'Kurtosis Norm', (-3, 3), '03_reverse_kurtosis_norm',path_plots)
    metric_plot(iso_ratio, 'Isotropy Ratio', (1.0, 2.0), '04_reverse_isotropy_ratio',path_plots, smoothed = True, window_size = 5,)
    metric_plot(frobenius_norm_difference, 'Frobenius_norm_difference', (-0.1, 1.25), '05_reverse_forbenius_norm', path_plots)
    metric_plot(iso_values, 'Isotropy Values',None ,'06_reverse_isotropy_values', path_plots)
'''