"""Training entry point for standard DDPM (constant weighting)."""

from __future__ import annotations

from argparse import Namespace
from typing import Optional, Sequence

from ._shared import parse_with_weighting, run_with_method


def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    """Parse arguments with DDPM-specific defaults."""
    return parse_with_weighting('constant', argv)


def run(argv: Optional[Sequence[str]] = None) -> None:
    """Execute a DDPM training run."""
    args = parse_args(argv)
    run_with_method("ddpm", args)
