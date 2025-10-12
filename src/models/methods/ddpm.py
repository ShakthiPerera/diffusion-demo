"""Model builder for baseline DDPM."""

from __future__ import annotations

import argparse

import torch

from ..ddpm import GenericDDPM


def build_model(
    args: argparse.Namespace,
    device: torch.device,
    betas: torch.Tensor,
    eps_model: torch.nn.Module,
    reg_strength: float,
) -> GenericDDPM:
    """Return the standard GenericDDPM instance."""
    return GenericDDPM(
        eps_model=eps_model,
        betas=betas,
        criterion=args.criterion,
        lr=args.lr,
        ema_decay=args.ema_decay,
        device=device,
        reg_strength=reg_strength,
        reg_type=args.reg_type,
        snr_gamma=args.snr_gamma,
    )
