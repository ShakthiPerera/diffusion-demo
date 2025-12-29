"""Loss helpers for DDPM and ISO-DDPM."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

LossStats = Dict[str, torch.Tensor]


def iso_regulariser(eps_pred: torch.Tensor) -> torch.Tensor:
    squared_norm = eps_pred.square().mean(dim=0).mean() - 1.0
    return squared_norm.square()


def diffusion_loss(
    pred_noise: torch.Tensor,
    true_noise: torch.Tensor,
    alphas_cumprod_t: torch.Tensor,
    reg_strength: float,
) -> Tuple[torch.Tensor, LossStats]:
    mse = (pred_noise - true_noise) ** 2
    mse_scalar = mse.mean(dim=1)
    simple_mse = mse_scalar.mean()

    iso_loss = iso_regulariser(pred_noise)
    reg_loss = reg_strength * iso_loss

    total_loss = simple_mse + reg_loss
    stats: LossStats = {
        "simple_loss": simple_mse.detach(),
        "reg_loss": reg_loss.detach(),
    }
    return total_loss, stats
