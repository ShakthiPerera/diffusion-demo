"""Loss helpers for DDPM and ISO-DDPM."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

LossStats = Dict[str, torch.Tensor]


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
    reg_strength: float,
) -> Tuple[torch.Tensor, LossStats]:
    mse = (pred_noise - true_noise) ** 2
    mse_scalar = mse.mean(dim=1)
    simple_mse = mse_scalar.mean()

    iso_loss = iso_regulariser(pred_noise, reduction="mean")
    reg_loss = reg_strength * iso_loss

    total_loss = simple_mse + reg_loss
    stats: LossStats = {
        "simple_loss": simple_mse.detach(),
        "reg_loss": reg_loss.detach(),
    }
    return total_loss, stats
