"""Shared helpers for training method wrappers."""

from __future__ import annotations

from typing import Optional, Sequence

from ..train import build_parser, main as train_main
from ...models.methods import get_model_builder


def parse_with_weighting(weighting: str, argv: Optional[Sequence[str]] = None):
    """Return parsed arguments with a default weighting."""
    parser = build_parser()
    parser.set_defaults(weighting=weighting)
    return parser.parse_args(argv)


def run_with_method(method_name: str, args) -> None:
    """Dispatch training using the specified method and already-parsed args."""
    model_builder = get_model_builder(method_name)
    train_main(args, model_builder=model_builder)
