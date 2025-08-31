'''Denoising diffusion model.'''

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
    Plain vanilla DDPM module.

    Summary
    -------
    This module establishes a plain vanilla DDPM variant.
    It is basically a container and wrapper for an
    epsilon model and for the scheduling parameters.
    The class provides methods implementing the forward
    and reverse diffusion processes, respectively.
    Also, the stochastic loss can be computed for training.

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
        Regularization strength.
    reg_type : str
        Type of regularization ('iso', 'mean_l2', 'var_l2', 'skew', 'kurt', 'var_mi', 'kl', 'mmd_rbf', 'mmd_linear').
    '''

    def __init__(self,
                 eps_model,
                 betas,
                 criterion='mse',
                 lr=1e-04,
                 reg=0.01,
                 reg_type='iso'):
        super().__init__()

        # set trainable epsilon model
        self.eps_model = eps_model

        # set loss function criterion
        if criterion == 'mse':
            self.criterion = nn.MSELoss(reduction='mean')
        elif criterion == 'mae':
            self.criterion = nn.L1Loss(reduction='mean')
        elif callable(criterion):
            self.criterion = criterion
        else:
            raise ValueError('Criterion could not be determined')

        # set initial learning rate
        self.lr = abs(lr)

        # set arrays for iso_difference, eps_pred and eps
        self.iso_difference_list = []
        self.eps_pred_list = []
        self.eps_list = []

        self.reg = reg
        self.reg_type = reg_type

        # to save losses
        self.train_losses = []
        self.val_losses = []

        self.norms = []

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # set scheduling parameters
        betas = torch.as_tensor(betas).view(-1)  # note that betas[0] corresponds to t = 1.0

        if betas.min() <= 0 or betas.max() >= 1:
            raise ValueError('Invalid beta values encountered')

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        betas_tilde = (1 - alphas_bar[:-1]) / (1 - alphas_bar[1:]) * betas[1:]
        betas_tilde = nn.functional.pad(betas_tilde, pad=(1, 0), value=0.0)  # ensure betas_tilde[0] = 0.0

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('betas_tilde', betas_tilde)

    def compute_snr(self, tids):
        # assuming α_t = alphas_bar[tids], σ_t² = 1 − ᾱ
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

    def denoise_step(self, x, tids, random_sample=False):
        '''Perform single reverse process step.'''
        tids = torch.as_tensor(tids, device=x.device).view(-1, 1)  # ensure (batch_size>=1, 1)-shaped tensor
        ts = tids.to(x.dtype) + 1  # note that tidx = 0 corresponds to t = 1.0

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
        else:
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
        isotropy = []
        for tidx in reversed(range(self.num_steps)):
            if tidx > 0:
                x_denoised = self.denoise_step(x_denoised, tidx, random_sample=True)
                iso = self.isotropy(x_denoised)
                isotropy.append(iso)
            else:
                x_denoised, _ = self.denoise_step(x_denoised, tidx, random_sample=False)
                iso = self.isotropy(x_denoised)
                isotropy.append(iso)

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

        # simple mse
        mse = self.criterion(eps_pred, eps)

        # compute weights for min_snr
        snr = self.compute_snr(tids.squeeze(-1))
        gamma = getattr(self, 'snr_gamma', 5.0)
        base_w = torch.minimum(snr, torch.tensor(gamma, device=snr.device)) / snr
        mse_weight = base_w.view(-1, *([1]*(eps_pred.ndim-1)))
        mse_weighted = (mse * mse_weight.squeeze(-1)).mean()

        simple_mse = mse.mean()

        # regularizer based on reg_type
        dim = float(eps.shape[-1])

        if self.reg_type == 'iso':
            squared_norm_preds = torch.mean(torch.sum(eps_pred**2, dim=tuple(range(1, eps_pred.ndim)))) / dim
            squared_norm_true = torch.tensor(1.0, device=eps.device)
            norm_loss = self.criterion(squared_norm_preds, squared_norm_true)
        elif self.reg_type == 'mean_l2':
            mean_pred = eps_pred.mean(dim=0).mean()  # Mean over batch for each feature
            mean_true = torch.tensor(0.0, device=eps.device)  # Gaussian mean = 0
            norm_loss = self.criterion(mean_pred, mean_true)
        elif self.reg_type == 'var_l2':
            var_pred = eps_pred.var(dim=0).mean()  # Variance over batch for each feature
            var_true = torch.tensor(1.0, device=eps.device)  # Gaussian variance = 1
            norm_loss = self.criterion(var_pred, var_true)
        elif self.reg_type == 'skew':
            def comp_skew(x):
                mu = x.mean(dim=0)
                std = x.std(dim=0) + 1e-8
                return ((x - mu)**3).mean(dim=0) / std**3
            skew_p = comp_skew(eps_pred)
            skew_t = torch.zeros_like(skew_p)  # Gaussian skewness = 0
            norm_loss = self.criterion(skew_p, skew_t)
        elif self.reg_type == 'kurt':
            def comp_kurt(x):
                mu = x.mean(dim=0)
                std = x.std(dim=0) + 1e-8
                return ((x - mu)**4).mean(dim=0) / std**4 - 3
            kurt_p = comp_kurt(eps_pred)
            kurt_t = torch.zeros_like(kurt_p)  # Gaussian excess kurtosis = 0
            norm_loss = self.criterion(kurt_p, kurt_t)
        elif self.reg_type == 'var_mi':
            mean_p = eps_pred.mean(dim=-1)  # Mean over features for each sample
            mean_t = eps.mean(dim=-1)
            var_p = mean_p.var()  # Variance over batch
            var_t = mean_t.var()
            norm_loss = self.criterion(var_p, var_t)
        elif self.reg_type == 'kl':
            d = eps.shape[-1]
            mu_p = eps_pred.mean(dim=0)
            cov_p = torch.cov(eps_pred.t()) + 1e-6 * torch.eye(d, device=eps.device)
            mu_t = torch.zeros(d, device=eps.device)  # Gaussian mean = 0
            cov_t = torch.eye(d, device=eps.device)  # Gaussian covariance = I
            inv_cov_t = cov_t  # Identity matrix
            trace_term = torch.trace(inv_cov_t @ cov_p)
            quad_term = (mu_t - mu_p).unsqueeze(0) @ inv_cov_t @ (mu_t - mu_p).unsqueeze(1)
            logdet_term = torch.log(torch.det(cov_t) / torch.det(cov_p) + 1e-8)
            norm_loss = 0.5 * (trace_term + quad_term.squeeze() - d + logdet_term)
        elif self.reg_type in ['mmd_rbf', 'mmd_linear']:
            kernel = self.reg_type.split('_')[1]
            gamma = 1.0 / eps.shape[-1] if kernel == 'rbf' else None
            m, n = eps_pred.shape[0], eps.shape[0]
            if kernel == 'linear':
                Kxx = torch.mm(eps_pred, eps_pred.t())
                Kyy = torch.mm(eps, eps.t())
                Kxy = torch.mm(eps_pred, eps.t())
            elif kernel == 'rbf':
                xx = torch.sum(eps_pred * eps_pred, dim=1).unsqueeze(1)
                yy = torch.sum(eps * eps, dim=1).unsqueeze(1)
                Kxx = torch.exp(-gamma * (xx + xx.t() - 2 * torch.mm(eps_pred, eps_pred.t())))
                Kyy = torch.exp(-gamma * (yy + yy.t() - 2 * torch.mm(eps, eps.t())))
                Kxy = torch.exp(-gamma * (xx + yy.t() - 2 * torch.mm(eps_pred, eps.t())))
            norm_loss = Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2 * Kxy.sum() / (m * n)
        else:
            raise ValueError(f'Unknown reg_type: {self.reg_type}')

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
        self.optimizer.step()
        return loss.item(), simple_diff_loss.item(), norm_loss.item()

    def get_eps_pred_list(self):
        return self.eps_pred_list

    def get_eps_list(self):
        return self.eps_list

    def get_iso_difference_list(self):
        return self.iso_difference_list

    def validate(self, val_loader):
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val_batch in val_loader:
                loss, _, _ = self.loss(x_val_batch[0]).item()
                val_loss += loss
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss