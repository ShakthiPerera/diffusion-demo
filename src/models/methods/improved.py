"""Placeholder model builder for Improved DDPM."""

from __future__ import annotations

import argparse
from typing import NoReturn

import torch


def build_model(
    args: argparse.Namespace,
    device: torch.device,
    betas: torch.Tensor,
    eps_model: torch.nn.Module,
    reg_strength: float,
) -> NoReturn:
    """Placeholder until Improved DDPM model is implemented."""
    raise NotImplementedError(
        "Improved DDPM model builder is not yet implemented. "
        "Add the appropriate model construction logic in src/models/methods/improved.py."
    )
