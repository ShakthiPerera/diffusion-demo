import random
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .unet import UNet
from .schedules import make_beta_schedule
import torch.nn.functional as F

class DDPM(pl.LightningModule):
    '''
    Plain vanilla DDPM module with EMA for stability.

    Summary
    -------
    This module establishes a DDPM variant with exponential moving average (EMA)
    for stable training and generation. It includes gradient clipping and supports
    learning rate scheduling. The class provides methods for forward and reverse
    diffusion processes and computes stochastic loss for training. The EMA model
    is used for inference in denoise_step and generate methods.

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

        # Initialize EMA model
        self.ema_decay = 0.995  # Adjusted for 40,000 steps
        self.ema_model = type(eps_model)([2, 128, 128, 128, 2], activation="relu", embed_dim=12)
        self.ema_model.load_state_dict(self.eps_model.state_dict())
        self.ema_model.eval()  # EMA model is only for inference

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

        # To save losses
        self.train_losses = []
        self.val_losses = []
        self.norms = []

        # Optimizer with AdamW
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

        # Set scheduling parameters
        betas = torch.as_tensor(betas).view(-1)
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
        else:
            return x_noisy

    def denoise_step(self, x, tids, random_sample=False, use_ema=False):
        '''Perform single reverse process step, optionally using EMA model.'''
        tids = torch.as_tensor(tids, device=x.device).view(-1, 1)
        ts = tids.to(x.dtype) + 1
        # Use EMA model for inference if specified, else use trainable model
        model = self.ema_model if use_ema else self.eps_model
        eps_pred = model(x, ts)
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
        else:
            return x_denoised_mean, x_denoised_var

    @torch.no_grad()
    def denoise_all_steps(self, xT):
        '''Perform and return all reverse process steps using EMA model.'''
        x_denoised = torch.zeros(self.num_steps + 1, *(xT.shape), device=xT.device)
        x_denoised[0] = xT
        for idx, tidx in enumerate(reversed(range(self.num_steps))):
            if tidx > 0:
                x_denoised[idx + 1] = self.denoise_step(x_denoised[idx], tidx, random_sample=True, use_ema=True)
            else:
                x_denoised[idx + 1], _ = self.denoise_step(x_denoised[idx], tidx, random_sample=False, use_ema=True)
        return x_denoised

    @torch.no_grad()
    def generate(self, sample_shape, num_samples=1):
        '''Generate random samples through the reverse process using EMA model.'''
        x_denoised = torch.randn(num_samples, *sample_shape, device=self.device)
        isotropy = []
        for tidx in reversed(range(self.num_steps)):
            if tidx > 0:
                x_denoised = self.denoise_step(x_denoised, tidx, random_sample=True, use_ema=True)
                
            else:
                x_denoised, _ = self.denoise_step(x_denoised, tidx, random_sample=False, use_ema=True)
        return x_denoised, isotropy

    def isotropy(self, data):
        data = data.detach().cpu().numpy()
        iso = np.vdot(data, data) / len(data)
        return iso

    def loss(self, x, loss_weighting_type='constant'):
        tids = torch.randint(0, self.num_steps, (x.shape[0], 1), device=x.device)
        ts = tids.to(x.dtype) + 1
        x_noisy, eps = self.diffuse(x, tids, return_eps=True)
        eps_pred = self.eps_model(x_noisy, ts)
        mse = self.criterion(eps_pred, eps)
        snr = self.compute_snr(tids.squeeze(-1).squeeze(0))
        gamma = getattr(self, 'snr_gamma', 5.0)
        base_w = torch.minimum(snr, torch.tensor(gamma, device=snr.device)) / snr
        mse_weight = base_w.view(-1, *([1]*(eps_pred.ndim-1)))
        mse_weighted = (mse * mse_weight.squeeze(-1)).mean()
        simple_mse = mse.mean()
        squared_norm_preds = torch.mean(torch.sum(eps_pred**2, dim=1)) / 2.0
        norm_loss = self.criterion(squared_norm_preds.to(eps_pred.device),
                                  torch.tensor(1.0, device=eps_pred.device))
        reg_loss = self.reg * norm_loss
        if loss_weighting_type == 'min_snr':
            loss = mse_weighted + reg_loss
        elif loss_weighting_type == 'constant':
            loss = simple_mse + reg_loss
        return loss, simple_mse, norm_loss

    def train_step(self, x_batch, loss_weighting_type):
        self.optimizer.zero_grad()
        loss, simple_diff_loss, norm_loss = self.loss(x_batch, loss_weighting_type)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.5)  # Gradient clipping
        self.optimizer.step()
        # Update EMA weights
        for param, ema_param in zip(self.eps_model.parameters(), self.ema_model.parameters()):
            ema_param.data = self.ema_decay * ema_param.data + (1 - self.ema_decay) * param.data
        return loss.item(), simple_diff_loss.item(), norm_loss.item()

    def validate(self, val_loader):
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val_batch in val_loader:
                x_val_batch = x_val_batch[0].to(self.device)  # Unpack TensorDataset
                loss, _, _ = self.loss(x_val_batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss