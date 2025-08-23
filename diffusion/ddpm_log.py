'''Denoising diffusion model.'''

import random
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import csv
import os

from .unet import UNet
from .schedules import make_beta_schedule
import torch.nn.functional as F

class DDPM(pl.LightningModule):
    '''
    Plain vanilla DDPM module with L2 norm logging.

    Summary
    -------
    This module establishes a plain vanilla DDPM variant.
    It computes the L2 norm of the predicted noise across the feature dimension
    (size 2) and averages across the batch in the loss function, saving these
    norms to a CSV file during training.

    Parameters
    ----------
    eps_model : PyTorch module
        Trainable noise-predicting model.
    betas : array-like
        Beta parameter schedule.
    criterion : {'mse', 'mae'} or callable
        Loss function criterion.
    lr : float
        Optimizer learning rate.
    reg : float
        Regularization strength for norm loss.
    '''

    def __init__(self,
                 eps_model,
                 betas,
                 criterion='mse',
                 lr=1e-04,
                 reg=0.01):
        super().__init__()

        # Set trainable epsilon model
        self.eps_model = eps_model

        # Set loss function criterion
        if criterion == 'mse':
            self.criterion = nn.MSELoss(reduction='mean')
        elif criterion == 'mae':
            self.criterion = nn.L1Loss(reduction='mean')
        elif callable(criterion):
            self.criterion = criterion
        else:
            raise ValueError('Criterion could not be determined')

        # Set initial learning rate
        self.lr = abs(lr)
        self.reg = reg

        # Lists to store metrics
        self.eps_pred_list = []
        self.eps_list = []
        self.train_losses = []
        self.val_losses = []
        self.norms = []  # Store (step, norm) tuples

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Set scheduling parameters
        betas = torch.as_tensor(betas).view(-1)  # betas[0] corresponds to t = 1.0
        if betas.min() <= 0 or betas.max() >= 1:
            raise ValueError('Invalid beta values encountered')

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        betas_tilde = (1 - alphas_bar[:-1]) / (1 - alphas_bar[1:]) * betas[1:]
        betas_tilde = nn.functional.pad(betas_tilde, pad=(1, 0), value=0.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('betas_tilde', betas_tilde)

    def compute_snr(self, tids):
        α_bar = self.alphas_bar[tids]
        sigma2 = (1 - α_bar)
        snr = α_bar / sigma2
        return snr

    @property
    def num_steps(self):
        '''Get the total number of time steps.'''
        return len(self.betas)

    def forward(self, x, t):
        '''Run the noise-predicting model.'''
        return self.eps_model(x, t)

    def diffuse_step(self, x, tidx):
        '''Simulate single forward process step.'''
        beta = self.betas[tidx]
        eps = torch.randn_like(x)
        x_noisy = (1 - beta).sqrt() * x + beta.sqrt() * eps
        return x_noisy

    def diffuse_all_steps_till_time(self, x0, time):
        '''Simulate and return all forward process steps.'''
        x_noisy = torch.zeros(time + 1, *x0.shape, device=x0.device)
        x_noisy[0] = x0
        for tidx in range(time):
            x_noisy[tidx + 1] = self.diffuse_step(x_noisy[tidx], tidx)
        return x_noisy

    def diffuse_all_steps(self, x0):
        '''Simulate and return all forward process steps.'''
        x_noisy = torch.zeros(self.num_steps + 1, *x0.shape, device=x0.device)
        x_noisy[0] = x0
        for tidx in range(self.num_steps):
            x_noisy[tidx + 1] = self.diffuse_step(x_noisy[tidx], tidx)
        return x_noisy

    def diffuse(self, x0, tids, return_eps=False):
        '''Simulate multiple forward steps at once.'''
        alpha_bar = self.alphas_bar[tids]
        eps = torch.randn_like(x0)
        missing_shape = [1] * (eps.ndim - alpha_bar.ndim)
        alpha_bar = alpha_bar.view(*alpha_bar.shape, *missing_shape)
        x_noisy = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps
        if return_eps:
            return x_noisy, eps
        return x_noisy

    def denoise_step(self, x, tids, random_sample=False):
        '''Perform single reverse process step.'''
        tids = torch.as_tensor(tids, device=x.device).view(-1, 1)
        ts = tids.to(x.dtype) + 1
        eps_pred = self.eps_model(x, ts)
        p = 1 / self.alphas[tids].sqrt()
        q = self.betas[tids] / (1 - self.alphas_bar[tids]).sqrt()
        missing_shape = [1] * (eps_pred.ndim - ts.ndim)
        p = p.view(*p.shape, *missing_shape)
        q = q.view(*q.shape, *missing_shape)
        x_denoised_mean = p * (x - q * eps_pred)
        x_denoised_var = self.betas_tilde[tids]
        if random_sample:
            eps = torch.randn_like(x_denoised_mean)
            x_denoised = x_denoised_mean + x_denoised_var.sqrt() * eps
            return x_denoised
        return x_denoised_mean, x_denoised_var

    @torch.no_grad()
    def denoise_all_steps(self, xT):
        '''Perform and return all reverse process steps.'''
        x_denoised = torch.zeros(self.num_steps + 1, *(xT.shape), device=xT.device)
        x_denoised[0] = xT
        for idx, tidx in enumerate(reversed(range(self.num_steps))):
            if tidx > 0:
                x_denoised[idx + 1] = self.denoise_step(x_denoised[idx], tidx, random_sample=True)
            else:
                x_denoised[idx + 1], _ = self.denoise_step(x_denoised[idx], tidx, random_sample=False)
        return x_denoised

    @torch.no_grad()
    def generate(self, sample_shape, num_samples=1):
        '''Generate random samples through the reverse process.'''
        x_denoised = torch.randn(num_samples, *sample_shape, device=self.device)
        for tidx in reversed(range(self.num_steps)):
            if tidx > 0:
                x_denoised = self.denoise_step(x_denoised, tidx, random_sample=True)
            else:
                x_denoised, _ = self.denoise_step(x_denoised, tidx, random_sample=False)
        return x_denoised, []

    def loss(self, x, loss_weighting_type='constant'):
        '''Compute stochastic loss with L2 norm of predicted noise.'''
        tids = torch.randint(0, self.num_steps, (x.shape[0], 1), device=x.device)
        ts = tids.to(x.dtype) + 1
        x_noisy, eps = self.diffuse(x, tids, return_eps=True)
        eps_pred = self.eps_model(x_noisy, ts)

        # Simple MSE
        mse = self.criterion(eps_pred, eps)

        # Compute weights
        snr = self.compute_snr(tids.squeeze(-1))
        gamma = getattr(self, 'snr_gamma', 5.0)
        base_w = torch.minimum(snr, torch.tensor(gamma, device=snr.device)) / snr
        mse_weight = base_w.view(-1, *([1]*(eps_pred.ndim-1)))
        mse_weighted = (mse * mse_weight.squeeze(-1)).mean()
        simple_mse = mse.mean()

        # Norm regularization
        squared_norm_preds = torch.mean(torch.sum(eps_pred**2, dim=1)) / 2.0
        norm_loss = self.criterion(squared_norm_preds, torch.tensor(1.0, device=eps_pred.device))
        reg_loss = self.reg * norm_loss

        # Compute L2 norm for logging
        norm_preds = torch.linalg.norm(eps_pred, ord=2, dim=1)  # L2 norm across dim=1 (size 2)
        avg_norm = torch.mean(norm_preds)  # Mean across batch

        if loss_weighting_type == 'min_snr':
            loss = mse_weighted + reg_loss
        elif loss_weighting_type == 'constant':
            loss = simple_mse + reg_loss
        return loss, simple_mse, squared_norm_preds, avg_norm

    def train_step(self, x_batch, loss_weighting_type, step, csv_path=None, save_interval=10):
        '''Perform a single training step and log L2 norm.'''
        self.optimizer.zero_grad()
        loss, simple_diff_loss, iso_val, avg_norm = self.loss(x_batch, loss_weighting_type)
        loss.backward()
        self.optimizer.step()

        # Store and save norm every save_interval steps
        if csv_path and step % save_interval == 0:
            self.norms.append((step, avg_norm.item(), iso_val.item()))
            self.save_metrics_to_csv(csv_path)

        return loss.item(), simple_diff_loss.item(), iso_val.item(), avg_norm.item()

    def save_metrics_to_csv(self, csv_path):
        '''Save L2 norm data to CSV.'''
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Predicted_Noise_L2Norm", "IsoValue"])
            for step, norm, iso in self.norms:
                writer.writerow([step, f"{norm:.6f}", f"{iso:.6f}"])

    def get_eps_pred_list(self):
        return self.eps_pred_list

    def get_eps_list(self):
        return self.eps_list

    def get_iso_difference_list(self):
        return []

    def validate(self, val_loader):
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val_batch in val_loader:
                loss, _, _, _ = self.loss(x_val_batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss