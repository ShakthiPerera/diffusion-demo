"""Beta schedule utilities for diffusion models."""

from __future__ import annotations

import torch


def make_beta_schedule(
    num_steps: int,
    mode: str = "cosine",
    beta_range: tuple[float, float] = (1e-4, 0.02),
    cosine_s: float = 0.008,
) -> torch.Tensor:
    """Create a sequence of betas for the diffusion process."""
    mode = mode.lower()
    if mode == "linear":
        beta_start, beta_end = beta_range
        betas = torch.linspace(beta_start, beta_end, steps=num_steps)
    elif mode == "quadratic":
        beta_start, beta_end = beta_range
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, steps=num_steps) ** 2
    elif mode == "cosine":
        s = abs(cosine_s)
        ts = torch.arange(num_steps + 1, dtype=torch.float32)
        alphas_bar = torch.cos(((ts / num_steps) + s) / (1.0 + s) * torch.pi / 2.0) ** 2
        alphas_bar = alphas_bar / alphas_bar.max()
        betas = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
        betas = torch.clip(betas, 1e-4, 0.999)
    else:
        raise ValueError(f"Unknown schedule type '{mode}'.")
    return betas.float()
