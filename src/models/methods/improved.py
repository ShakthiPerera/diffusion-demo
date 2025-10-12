"""Model builder for Improved DDPM with learned variances."""

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
    """Return a GenericDDPM configured for Improved DDPM training."""
    learn_sigma = bool(getattr(args, "learn_sigma", True))
    vlb_weight = float(getattr(args, "vlb_weight", 1e-3))
    variance_type = getattr(args, "variance_type", "learned_range")
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
        learn_sigma=learn_sigma,
        vlb_weight=vlb_weight,
        variance_type=variance_type,
    )
