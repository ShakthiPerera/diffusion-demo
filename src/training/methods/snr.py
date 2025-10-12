"""Training entry point for DDPM with SNR weighting."""

from __future__ import annotations

from argparse import Namespace
from typing import Optional, Sequence

from ._shared import parse_with_weighting, run_with_method


def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    """Parse arguments with SNR-specific defaults."""
    return parse_with_weighting('snr', argv)


def run(argv: Optional[Sequence[str]] = None) -> None:
    """Execute an SNR-weighted DDPM training run."""
    args = parse_args(argv)
    run_with_method("snr", args)
