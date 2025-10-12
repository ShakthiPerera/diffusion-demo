"""
Beta schedule utilities for diffusion models.

This module exposes a single function :func:`make_beta_schedule` which
constructs a sequence of beta values for a diffusion process.  The function
closely follows the implementation found in the original codebase while
remaining self‑contained.

Beta schedules determine how much Gaussian noise is added at each
time step during the diffusion process.  Four modes are supported:

``linear``
    Linearly interpolate between ``beta_range[0]`` and ``beta_range[1]``.
``quadratic``
    Quadratically interpolate between the square roots of the beta range and
    then square the result.
``cosine``
    Use the cosine schedule proposed in [1], computing betas from a smooth
    schedule on the cumulative product of alphas.
``sigmoid``
    Use a sigmoid schedule on the square root of the cumulative product of
    alphas.

The return value is a one‑dimensional ``torch.Tensor`` whose length is
``num_steps``.  Each entry lies in the range (0, 1).

References
----------

[1] Nichol, A. Q., & Dhariwal, P. (2021). Improved Denoising Diffusion
    Probabilistic Models. arXiv preprint arXiv:2102.09672.
"""

from __future__ import annotations

import torch


def make_beta_schedule(num_steps: int, mode: str = 'cosine',
                       beta_range: tuple[float, float] = (1e-4, 0.02),
                       cosine_s: float = 0.008,
                       sigmoid_range: tuple[float, float] = (-5.0, 5.0)) -> torch.Tensor:
    """Create a sequence of beta values for a diffusion process.

    Parameters
    ----------
    num_steps : int
        Number of diffusion steps (the length of the returned tensor).
    mode : {'linear', 'quadratic', 'cosine', 'sigmoid'}, optional
        The type of schedule to construct.  Default is ``'cosine'``.
    beta_range : tuple of two floats, optional
        Lower and upper bounds for ``linear`` and ``quadratic`` schedules.
    cosine_s : float, optional
        Offset parameter for the cosine schedule [1].
    sigmoid_range : tuple of two floats, optional
        Range of input values for the sigmoid schedule on ``sqrt(alpha_bar)``.

    Returns
    -------
    torch.Tensor of shape (``num_steps``,)
        Sequence of beta values for each diffusion step.
    """
    mode = mode.lower()
    if mode == 'linear':
        beta_start, beta_end = beta_range
        if not (0.0 < beta_start < 1.0 and 0.0 < beta_end < 1.0):
            raise ValueError('Beta bounds must lie in (0, 1).')
        betas = torch.linspace(beta_start, beta_end, steps=num_steps)
    elif mode == 'quadratic':
        beta_start, beta_end = beta_range
        if not (0.0 < beta_start < 1.0 and 0.0 < beta_end < 1.0):
            raise ValueError('Beta bounds must lie in (0, 1).')
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, steps=num_steps)**2
    elif mode == 'cosine':
        # cosine schedule on alpha_bar
        s = abs(cosine_s)
        ts = torch.arange(num_steps + 1, dtype=torch.float32)
        alphas_bar = torch.cos(((ts / num_steps) + s) / (1.0 + s) * torch.pi / 2.0)**2
        alphas_bar = alphas_bar / alphas_bar.max()
        betas = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
        betas = torch.clip(betas, 1e-4, 0.999)
    elif mode == 'sigmoid':
        # sigmoid schedule on sqrt(alpha_bar)
        start, end = sigmoid_range
        ts = torch.linspace(start, end, steps=num_steps + 1)
        sqrt_alpha_bar = torch.sigmoid(-ts)
        alphas_bar = sqrt_alpha_bar**2
        betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = torch.clip(betas, 1e-4, 0.999)
    else:
        raise ValueError(f"Unknown schedule type '{mode}'.")
    return betas.float()
