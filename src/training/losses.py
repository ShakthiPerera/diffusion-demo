"""
Utility functions for diffusion training losses.

This module centralises the loss computations used by the training loop
to make it easier to plug in alternative objectives (e.g. SNR weighting,
ISO variants) without modifying the core diffusion model.  The helpers
mirror the behaviour that previously lived inside ``GenericDDPM``.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch


LossStats = Dict[str, torch.Tensor]
ISO_REG_TYPES = {"iso", "iso_frob", "iso_split", "iso_logeig", "iso_bures"}


def compute_snr(alphas_cumprod_t: torch.Tensor) -> torch.Tensor:
    """Compute per-sample signal-to-noise ratios."""
    return alphas_cumprod_t / (1.0 - alphas_cumprod_t + 1e-8)


def snr_weights(snr: torch.Tensor, gamma: float) -> torch.Tensor:
    """Return the min-SNR weighting factors proposed in Improved DDPMs."""
    gamma_tensor = torch.full_like(snr, gamma)
    return torch.minimum(snr, gamma_tensor) / snr


def diffusion_loss(
    pred_noise: torch.Tensor,
    true_noise: torch.Tensor,
    alphas_cumprod_t: torch.Tensor,
    weighting: str,
    snr_gamma: float,
    reg_type: str,
    reg_loss_fn: Callable[..., torch.Tensor],
) -> Tuple[torch.Tensor, LossStats]:
    """Compute the diffusion objective and optional regularisation."""
    mse = (pred_noise - true_noise) ** 2
    mse_scalar = mse.mean(dim=1)
    snr = compute_snr(alphas_cumprod_t)
    weights = snr_weights(snr, snr_gamma)
    weighted_mse = (mse_scalar * weights).mean()
    simple_mse = mse_scalar.mean()

    device = pred_noise.device
    reg_loss = torch.zeros(1, device=device)
    if reg_loss_fn is not None:
        if reg_type == "iso":
            iso_loss = reg_loss_fn(pred_noise, true_noise, reduction="none")
            iso_weights = weights if weighting == "snr" else torch.ones_like(weights)
            reg_loss = (iso_loss * iso_weights).mean()
        else:
            reg_loss = reg_loss_fn(pred_noise, true_noise, reduction="mean")

    if weighting == "snr":
        total_loss = weighted_mse + reg_loss
    else:
        total_loss = simple_mse + reg_loss

    stats: LossStats = {
        "simple_loss": simple_mse.detach(),
        "weighted_loss": weighted_mse.detach(),
        "reg_loss": reg_loss.detach(),
        "snr_weights": weights.detach(),
    }
    return total_loss, stats


def _zero_reg_loss(eps_pred: torch.Tensor, _eps_true: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    if reduction == "none":
        return torch.zeros(eps_pred.size(0), device=eps_pred.device, dtype=eps_pred.dtype)
    return torch.zeros((), device=eps_pred.device, dtype=eps_pred.dtype)


def _isotropy_statistics(eps: torch.Tensor) -> Dict[str, torch.Tensor]:
    batch_size, dim = eps.shape
    mean = eps.mean(dim=0, keepdim=True)
    centered = eps - mean
    cov = centered.transpose(0, 1) @ centered / float(batch_size)
    cov = 0.5 * (cov + cov.transpose(0, 1))
    squared_norm_per_sample = eps.pow(2).sum(dim=1) / float(dim)
    return {
        "cov": cov,
        "squared_norm_per_sample": squared_norm_per_sample,
    }


def _iso_family_loss(
    eps_pred: torch.Tensor,
    reg_type: str,
    reduction: str,
) -> torch.Tensor:
    stats = _isotropy_statistics(eps_pred)
    cov = stats["cov"]
    squared_norm = stats["squared_norm_per_sample"]
    if reg_type == "iso":
        iso_error = (squared_norm - 1.0) ** 2
        if reduction == "none":
            return iso_error
        if reduction == "sum":
            return iso_error.sum()
        if reduction == "mean":
            return iso_error.mean()
        raise ValueError(f"Unknown reduction '{reduction}'.")
    if reduction == "none":
        raise ValueError(f"Reduction 'none' not supported for reg_type '{reg_type}'.")
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    if reg_type == "iso_frob":
        diff = cov - eye
        return (diff * diff).sum()
    if reg_type == "iso_split":
        diag = torch.diagonal(cov)
        off = cov - torch.diag(diag)
        return off.pow(2).sum() + (diag.mean() - 1.0) ** 2
    jitter = 1e-5
    eigvals = torch.linalg.eigvalsh(cov + jitter * eye)
    if reg_type == "iso_logeig":
        return eigvals.log().pow(2).mean()
    if reg_type == "iso_bures":
        return (eigvals.sqrt() - 1.0).pow(2).mean()
    raise ValueError(f"Unknown ISO reg_type '{reg_type}'.")


def build_reg_loss_fn(
    reg_type: str,
    reg_strength: float,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[..., torch.Tensor]:
    reg_type = reg_type.lower()
    reg_strength = float(reg_strength)
    if reg_strength <= 0.0:
        return _zero_reg_loss

    if reg_type in ISO_REG_TYPES:
        def _iso_reg_loss(
            eps_pred: torch.Tensor,
            _eps_true: torch.Tensor,
            reduction: str = "mean",
        ) -> torch.Tensor:
            base = _iso_family_loss(eps_pred, reg_type=reg_type, reduction=reduction)
            return reg_strength * base
        return _iso_reg_loss

    if reg_type == "mean_l2":
        def _mean_l2(
            eps_pred: torch.Tensor,
            _eps_true: torch.Tensor,
            reduction: str = "mean",
        ) -> torch.Tensor:
            if reduction == "none":
                raise ValueError("Reduction 'none' not supported for mean_l2.")
            mean_pred = eps_pred.mean(dim=0).mean()
            mean_true = torch.tensor(0.0, device=eps_pred.device, dtype=eps_pred.dtype)
            return reg_strength * criterion(mean_pred, mean_true)
        return _mean_l2

    if reg_type == "var_l2":
        def _var_l2(
            eps_pred: torch.Tensor,
            _eps_true: torch.Tensor,
            reduction: str = "mean",
        ) -> torch.Tensor:
            if reduction == "none":
                raise ValueError("Reduction 'none' not supported for var_l2.")
            var_pred = eps_pred.var(dim=0).mean()
            var_true = torch.tensor(1.0, device=eps_pred.device, dtype=eps_pred.dtype)
            return reg_strength * criterion(var_pred, var_true)
        return _var_l2

    if reg_type == "skew":
        def _skew(
            eps_pred: torch.Tensor,
            _eps_true: torch.Tensor,
            reduction: str = "mean",
        ) -> torch.Tensor:
            if reduction == "none":
                raise ValueError("Reduction 'none' not supported for skew.")
            mu = eps_pred.mean(dim=0)
            std = eps_pred.std(dim=0) + 1e-8
            skew_pred = ((eps_pred - mu) ** 3).mean(dim=0) / (std ** 3)
            skew_true = torch.zeros_like(skew_pred)
            return reg_strength * criterion(skew_pred, skew_true)
        return _skew

    if reg_type == "kurt":
        def _kurt(
            eps_pred: torch.Tensor,
            _eps_true: torch.Tensor,
            reduction: str = "mean",
        ) -> torch.Tensor:
            if reduction == "none":
                raise ValueError("Reduction 'none' not supported for kurt.")
            mu = eps_pred.mean(dim=0)
            std = eps_pred.std(dim=0) + 1e-8
            kurt_pred = ((eps_pred - mu) ** 4).mean(dim=0) / (std ** 4) - 3.0
            kurt_true = torch.zeros_like(kurt_pred)
            return reg_strength * criterion(kurt_pred, kurt_true)
        return _kurt

    if reg_type == "var_mi":
        def _var_mi(
            eps_pred: torch.Tensor,
            eps_true: torch.Tensor,
            reduction: str = "mean",
        ) -> torch.Tensor:
            if reduction == "none":
                raise ValueError("Reduction 'none' not supported for var_mi.")
            mean_pred_samples = eps_pred.mean(dim=1)
            mean_true_samples = eps_true.mean(dim=1)
            var_pred = mean_pred_samples.var()
            var_true = mean_true_samples.var()
            return reg_strength * criterion(var_pred, var_true)
        return _var_mi

    if reg_type == "kl":
        def _kl(
            eps_pred: torch.Tensor,
            eps_true: torch.Tensor,
            reduction: str = "mean",
        ) -> torch.Tensor:
            if reduction == "none":
                raise ValueError("Reduction 'none' not supported for kl.")
            d = eps_true.shape[-1]
            mu_p = eps_pred.mean(dim=0)
            cov_p = torch.cov(eps_pred.t()) + 1e-6 * torch.eye(d, device=eps_pred.device, dtype=eps_pred.dtype)
            mu_t = torch.zeros(d, device=eps_pred.device, dtype=eps_pred.dtype)
            cov_t = torch.eye(d, device=eps_pred.device, dtype=eps_pred.dtype)
            inv_cov_t = cov_t
            trace_term = torch.trace(inv_cov_t @ cov_p)
            diff = (mu_t - mu_p).unsqueeze(0)
            quad_term = diff @ inv_cov_t @ diff.transpose(0, 1)
            logdet_term = torch.log(torch.det(cov_t) / torch.det(cov_p) + 1e-8)
            kl_val = 0.5 * (trace_term + quad_term.squeeze() - d + logdet_term)
            return reg_strength * kl_val
        return _kl

    if reg_type in {"mmd_linear", "mmd_rbf"}:
        kernel = reg_type.split("_")[1]

        def _mmd(
            eps_pred: torch.Tensor,
            eps_true: torch.Tensor,
            reduction: str = "mean",
        ) -> torch.Tensor:
            if reduction == "none":
                raise ValueError("Reduction 'none' not supported for MMD.")
            m, n = eps_pred.shape[0], eps_true.shape[0]
            if kernel == "linear":
                Kxx = eps_pred @ eps_pred.t()
                Kyy = eps_true @ eps_true.t()
                Kxy = eps_pred @ eps_true.t()
            else:
                gamma = 1.0 / eps_true.shape[-1]
                xx = (eps_pred ** 2).sum(dim=1).unsqueeze(1)
                yy = (eps_true ** 2).sum(dim=1).unsqueeze(1)
                Kxx = torch.exp(-gamma * (xx + xx.t() - 2 * (eps_pred @ eps_pred.t())))
                Kyy = torch.exp(-gamma * (yy + yy.t() - 2 * (eps_true @ eps_true.t())))
                Kxy = torch.exp(-gamma * (xx + yy.t() - 2 * (eps_pred @ eps_true.t())))
            mmd_val = Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2.0 * Kxy.sum() / (m * n)
            return reg_strength * mmd_val
        return _mmd

    raise ValueError(f"Unknown reg_type '{reg_type}'.")
