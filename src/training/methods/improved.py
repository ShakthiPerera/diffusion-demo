"""Training entry point for Improved DDPM with learned variances."""

from __future__ import annotations

from argparse import Namespace
from typing import Optional, Sequence

from ..train import build_parser
from ._shared import run_with_method


def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    """Parse arguments with Improved-DDPM-specific defaults."""
    parser = build_parser()
    parser.set_defaults(
        weighting='constant',
        learn_sigma=True,
        variance_type='learned_range',
        vlb_weight=1e-3,
    )
    return parser.parse_args(argv)


def run(argv: Optional[Sequence[str]] = None) -> None:
    """Execute an Improved DDPM training run."""
    args = parse_args(argv)
    if not getattr(args, "learn_sigma", True):
        args.learn_sigma = True
    run_with_method("improved", args)
