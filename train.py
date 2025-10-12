"""Entry-point wrapper for the training CLI."""

from src.training.train import main, parse_args


if __name__ == "__main__":
    args = parse_args()
    main(args)
