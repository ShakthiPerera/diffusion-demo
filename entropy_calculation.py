"""Run a training job (optional) and compute entropy on generated/true data."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, Sequence

import numpy as np

from src import entropy_utils as ent
import train as train_module


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train (optional) and compute entropy on 2D diffusion outputs.")
    parser.add_argument("--dataset", type=str, choices=train_module.DATASETS, default="moon_scatter")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Where train.py writes outputs.")
    parser.add_argument("--train", dest="do_train", action="store_true", help="Run training before entropy calc.")
    parser.add_argument("--no-train", dest="do_train", action="store_false", help="Skip training if outputs exist.")
    parser.set_defaults(do_train=True)
    # minimal training knobs
    parser.add_argument("--train_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg_strength", type=float, default=0.0)
    parser.add_argument("--weighting", type=str, choices=["constant", "snr"], default="constant")
    parser.add_argument("--num_diffusion_steps", type=int, default=1000)
    parser.add_argument("--schedule", type=str, choices=["cosine", "linear", "quadratic"], default="cosine")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--noise_level", type=float, default=0.1, help="Noise level for dataset generation.")
    # entropy params
    parser.add_argument("--methods", type=str, default="knn,kde", help="Comma list from {knn,kde}.")
    parser.add_argument("--k", type=int, default=5, help="k for kNN entropy.")
    parser.add_argument("--metric", type=str, default="chebyshev", help="Distance metric for kNN entropy.")
    parser.add_argument("--bandwidth", type=float, default=None, help="KDE bandwidth (None uses Scott's rule).")
    parser.add_argument("--base", type=float, default=np.e, help="Log base for entropy (np.e=nats, 2=bits).")
    return parser.parse_args(argv)


def run_training(args: argparse.Namespace) -> None:
    parser = train_module.build_parser()
    train_args = parser.parse_args([])
    # override a focused subset from entropy script args
    train_args.dataset = args.dataset
    train_args.train_steps = args.train_steps
    train_args.batch_size = args.batch_size
    train_args.lr = args.lr
    train_args.reg_strength = args.reg_strength
    train_args.weighting = args.weighting
    train_args.noise_level = args.noise_level
    train_args.num_diffusion_steps = args.num_diffusion_steps
    train_args.schedule = args.schedule
    train_args.save_dir = args.save_dir
    train_args.gpu_id = args.gpu_id
    # keep defaults for other options (num_samples, hidden_dim, etc.)
    train_module.train(train_args)


def compute_entropies(data: np.ndarray, methods: list[str], args: argparse.Namespace) -> Dict[str, float]:
    results: Dict[str, float] = {}
    if "knn" in methods:
        results["knn"] = ent.knn_entropy_kozachenko_leonenko(
            data,
            k=args.k,
            base=args.base,
            metric=args.metric,
        )
    if "kde" in methods:
        results["kde"] = ent.kde_entropy(
            data,
            bandwidth=args.bandwidth,
            base=args.base,
        )
    return results


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = os.path.join(args.save_dir, args.dataset)
    gen_path = os.path.join(run_dir, f"generated_{args.dataset}.npy")
    true_path = os.path.join(run_dir, f"true_{args.dataset}.npy")

    if args.do_train or not (os.path.exists(gen_path) and os.path.exists(true_path)):
        run_training(args)

    if not os.path.exists(gen_path) or not os.path.exists(true_path):
        raise FileNotFoundError(f"Expected generated/true data under {run_dir}. Run with --train to generate them.")

    generated = np.load(gen_path)
    true_data = np.load(true_path)
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

    gen_entropy = compute_entropies(generated, methods, args)
    true_entropy = compute_entropies(true_data, methods, args)

    summary: Dict[str, Any] = {
        "dataset": args.dataset,
        "methods": methods,
        "generated_entropy": gen_entropy,
        "true_entropy": true_entropy,
    }

    out_path = os.path.join(run_dir, f"entropy_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Entropy results saved to {out_path}")
    for label, res in [("generated", gen_entropy), ("true", true_entropy)]:
        print(f"{label}: " + ", ".join(f"{k}={v:.6f}" for k, v in res.items()))


if __name__ == "__main__":
    main()
