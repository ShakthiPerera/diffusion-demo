"""High-level entry point for launching different diffusion training methods."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src.training.methods import get_available_methods, get_method_runner


def parse_main_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse top-level arguments and return remaining args for the method runner."""
    available = get_available_methods()
    parser = argparse.ArgumentParser(
        description="Unified interface for running diffusion training variants.",
        add_help=True,
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=available,
        help="Select which training variant to run.",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="Print the available training methods and exit.",
    )
    args, remaining = parser.parse_known_args(argv)
    if args.list_methods:
        print("\n".join(available))
        parser.exit()
    if not args.method:
        parser.error("--method is required unless --list-methods is supplied.")
    return args, remaining


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch to the selected training method."""
    args, remaining = parse_main_args(argv)
    runner = get_method_runner(args.method)
    runner(remaining)


if __name__ == "__main__":
    main(sys.argv[1:])
