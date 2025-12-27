"""Loss helpers for DDPM and ISO-DDPM."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

LossStats = Dict[str, torch.Tensor]


def compute_snr(alphas_cumprod_t: torch.Tensor) -> torch.Tensor:
    return alphas_cumprod_t / (1.0 - alphas_cumprod_t + 1e-8)


def snr_weights(snr: torch.Tensor, gamma: float) -> torch.Tensor:
    gamma_tensor = torch.full_like(snr, gamma)
    return torch.minimum(snr, gamma_tensor) / snr


def iso_regulariser(eps_pred: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    squared_norm = eps_pred.pow(2).sum(dim=1) / float(eps_pred.shape[1])
    iso_error = (squared_norm - 1.0) ** 2
    if reduction == "none":
        return iso_error
    if reduction == "sum":
        return iso_error.sum()
    return iso_error.mean()


def diffusion_loss(
    pred_noise: torch.Tensor,
    true_noise: torch.Tensor,
    alphas_cumprod_t: torch.Tensor,
    weighting: str,
    snr_gamma: float,
    reg_strength: float,
) -> Tuple[torch.Tensor, LossStats]:
    mse = (pred_noise - true_noise) ** 2
    mse_scalar = mse.mean(dim=1)
    snr = compute_snr(alphas_cumprod_t)
    weights = snr_weights(snr, snr_gamma)
    weighted_mse = (mse_scalar * weights).mean()
    simple_mse = mse_scalar.mean()

    iso_loss = iso_regulariser(pred_noise, reduction="none")
    iso_weights = weights if weighting == "snr" else torch.ones_like(weights)
    reg_loss = reg_strength * (iso_loss * iso_weights).mean()

    total_loss = (weighted_mse if weighting == "snr" else simple_mse) + reg_loss
    stats: LossStats = {
        "simple_loss": simple_mse.detach(),
        "weighted_loss": weighted_mse.detach(),
        "reg_loss": reg_loss.detach(),
        "snr_weights": weights.detach(),
    }
    return total_loss, stats
