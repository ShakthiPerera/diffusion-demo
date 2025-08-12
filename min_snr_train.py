import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from prdc import compute_prdc  # precision, recall, density, coverage :contentReference[oaicite:8]{index=8}
from sklearn.datasets import make_swiss_roll, make_moons
from sklearn.preprocessing import MinMaxScaler
import os 
from tqdm import tqdm 

noise_sigma = 0.3
regs = [0,0.01,0.05,0.1,0.2,0.3,0.4,0.5]
methods = ['iso', 'min_snr']
dataset_name = 'central_banana'
batch_size = 1024

num = 0
device = f'cuda:{num}' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

#########################################################################################

# banana and central PDF

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def banana_cen_pdf(theta, max_distance, concentration_factor=2, decay=0.5):
    r = max_distance * (np.exp(-decay * theta) * (1 + concentration_factor * np.sin(theta)))
    return r

def generate_cen_banana_points(num_points, sigma=0.3, max_distance=1.0):
    X = []
    for _ in range(num_points):
        theta = np.random.exponential(0.25)
        r = banana_cen_pdf(theta, max_distance)
        width_effect = sigma * (1 - (r / max_distance))
        x_center = r * np.cos(theta)
        y_center = r * np.sin(theta)
        x_offset = np.random.uniform(-width_effect / 2, width_effect / 2)
        y_offset = np.random.uniform(-width_effect / 2, width_effect / 2)
        x_new = x_center + x_offset
        y_new = y_center + y_offset
        X.append((x_new, y_new))
    return np.array(X)

def generate_central_points(num_points, max_distance=0.5, sigma=0.1):
    central_points = []
    for _ in range(num_points):
        theta = np.pi / 3 + np.random.normal(0, 0.5)
        r = max_distance * (np.random.uniform(0.5, 0.8))
        x_center = r * np.cos(theta)
        y_center = r * np.sin(theta)
        x_offset = np.random.normal(0, sigma / 2)
        y_offset = np.random.normal(0, sigma / 2)
        central_points.append((x_center + x_offset, y_center + y_offset))
    return np.array(central_points)

def generate_cen_banana(total_points=10000, central_points=500):
    banana_points = generate_cen_banana_points(total_points - central_points)
    middle_points = generate_central_points(central_points)
    all_points = np.vstack((banana_points, middle_points))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_rescaled = scaler.fit_transform(all_points)
    return x_rescaled

#########################################################################################

def create_crescent_with_scatter(num_samples=10000, noise_level=0.1, random_state=42,
                                            crescent_class=0, scatter_band=0.3, scatter_density=500):
    # Generate the two-class moons dataset and filter for one crescent
    X, y = make_moons(2*num_samples - 2*scatter_density, noise=noise_level, random_state=random_state)
    X_single_crescent = X[y == crescent_class]

    np.random.seed(random_state)

    X_scatter = []
    for point in X_single_crescent:
        x_crescent, y_crescent = point

        # Add noise to the crescent points within a specified scatter band
        x_scattered = x_crescent + scatter_band * np.random.randn()
        y_scattered = y_crescent + scatter_band * np.random.randn()
        X_scatter.append((x_scattered, y_scattered))

    X_scatter = np.array(X_scatter[:scatter_density])  # Limit the scatter points to the specified density

    X_combined = np.vstack((X_single_crescent, X_scatter))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_crescent_normalized = scaler.fit_transform(X_combined)

    return X_crescent_normalized

#########################################################################################

def create_swiss_roll_samples(num_samples=10000, noise_level=0.5, random_state=42):
  X, _ = make_swiss_roll(num_samples, noise=noise_level, random_state=random_state)

  # Restrict to 2D
  X = X[:,[0,2]]

  # Normalize to be between -1 and 1
  scaler = MinMaxScaler(feature_range=(-1, 1))
  X_normalized = scaler.fit_transform(X)

  return X_normalized


# Prepare data
# X_normalized = generate_cen_banana(total_points=10000)
X_normalized = create_crescent_with_scatter(num_samples=10000)

X_train, X_val = train_test_split(X_normalized, test_size=0.2)

X_train = torch.tensor(X_train).float()
X_val = torch.tensor(X_val).float()

train_set = TensorDataset(X_train)
val_set = TensorDataset(X_val)

print('No. train images:', len(train_set))
print('No. val. images:', len(val_set))

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
    num_workers=min(4, os.cpu_count()),
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    drop_last=False,
    shuffle=False,
    num_workers=min(4, os.cpu_count()),
    pin_memory=True
)

from diffusion import ConditionalDenseModel, make_beta_schedule
from diffusion.ddpm_norms import DDPM as ddpm_iso

def create_model(reg=0.1):

    num_features = [2, 128, 128, 128, 2]

    eps_model = ConditionalDenseModel(num_features, activation='relu', embed_dim=12)

    betas = make_beta_schedule(num_steps=200, mode='linear', beta_range=(1e-04, 0.02))
    # betas = make_beta_schedule(num_steps=1000, mode='quadratic', beta_range=(1e-04, 0.02))
    # betas = make_beta_schedule(num_steps=1000, mode='cosine', cosine_s=0.008)
    # betas = make_beta_schedule(num_steps=1000, mode='sigmoid', sigmoid_range=(-5, 5))

    ddpm_model = ddpm_iso(eps_model=eps_model, betas=betas, criterion='mse', lr=1e-03, reg=reg)

    return ddpm_model

def train(ddpm, device, num_epochs, method):
    # Training loop

    train_losses = torch.zeros(num_epochs)
    diff_losses = torch.zeros(num_epochs)
    norm_losses = torch.zeros(num_epochs)

    ddpm.to(device)
    training_length = len(train_loader)

    for epoch in tqdm(range(num_epochs)):
        ddpm.train()
        loss, simple_diff_loss, norms = 0, 0, 0
        for x_batch in train_loader:
            x_batch = x_batch[0].to(device)
            loss_, simple_loss, norm_loss = ddpm.train_step(x_batch, method)
            loss += loss_
            simple_diff_loss += simple_loss
            norms += norm_loss

        loss = loss / training_length
        avg_diff_loss = simple_diff_loss / training_length
        avg_norm_loss = norms / training_length

        # print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss:.4f}')

        # Save losses
        train_losses[epoch] = loss
        diff_losses[epoch] = avg_diff_loss
        norm_losses[epoch] = avg_norm_loss

    return train_losses, diff_losses, norm_losses

results = []
for method in methods:
    for reg in regs:
        metrics = {'precision': [], 'recall': [], 'density': [], 'coverage': []}
        for run in range(3):
            ddpm = create_model(reg=reg)
            print(f"Started Training of model in {method} method with regularization of {reg} value for run {run+1}.")
            train(ddpm, device, 1000, method)
            fake, _ = ddpm.generate(sample_shape=X_train.shape[1:], num_samples=10000)
            fake_np = fake.cpu().numpy()
            prdc = compute_prdc(real_features=X_normalized, fake_features=fake_np, nearest_k=5)
            for key in metrics:
                metrics[key].append(prdc[key])
        # compute mean
        results.append({
            'method': method,
            'reg': reg,
            'precision': np.mean(metrics['precision']),
            'recall':    np.mean(metrics['recall']),
            'density':   np.mean(metrics['density']),
            'coverage':  np.mean(metrics['coverage'])
        })

df = pd.DataFrame(results)
df.to_csv('prdc_results_iso_vs_min_snr_scatter_moon.csv', index=False)
print("Saved mean PRDC results CSV")