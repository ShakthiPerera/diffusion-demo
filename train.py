"""
Training script for the unified 2D diffusion model.

This script ties together the dataset generator, beta schedule, diffusion
model, and evaluation metrics into a single command‑line program.  It
provides sensible defaults and can be used as a drop‑in replacement for the
proliferation of near‑duplicate training scripts found in the original
repository.  Only the PRDC metrics are computed at the end of training;
expensive and rarely used metrics such as MMD are intentionally omitted.

Example
-------

To train a model on the eight Gaussians dataset for 10 k optimisation steps:

```
python train.py --dataset eight_gaussians --train_steps 10000 --batch_size 512
```

After training the script will sample ``num_samples`` points and compute
precision, recall, density, and coverage against the training data.
"""

from __future__ import annotations

import argparse
import os

# When using CUDA >= 10.2 with deterministic algorithms enabled, a cuBLAS
# workspace configuration must be specified via this environment variable to
# avoid runtime errors.  See the PyTorch and CUDA documentation for
# details.  We set a small workspace by default.  Users can override this
# externally if desired.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use a non‑interactive backend suitable for scripts
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from datasets import Synthetic2DDataset
from schedules import make_beta_schedule
from ddpm import GenericDDPM
from dense_layers import ConditionalDenseModel
from metrics import compute_prdc


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description='Train a 2D DDPM model with unified components.')
    parser.add_argument('--dataset', type=str, default='eight_gaussians',
                        help=('Name of the synthetic dataset to use.  Supported values include '
                              'eight_gaussians, moons, swiss_roll, banana, central_banana, '
                              'moon_circles, banana_circles and moon_scatter.'))
    parser.add_argument('--num_samples', type=int, default=10_000,
                        help='Number of samples to generate for training and evaluation.')
    parser.add_argument('--noise_level', type=float, default=0.5,
                        help='Noise level for datasets that support it.')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--num_diffusion_steps', type=int, default=1000,
                        help='Number of diffusion steps (T).')
    parser.add_argument('--schedule', type=str, default='linear', choices=['linear', 'quadratic', 'cosine', 'sigmoid'],
                        help='Type of beta schedule to use.')
    parser.add_argument('--train_steps', type=int, default=50_000,
                        help='Number of optimisation steps.')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for Adam optimiser.')
    parser.add_argument('--criterion', type=str, default='mse', choices=['mse', 'l1'],
                        help='Loss criterion to use.')
    parser.add_argument('--weighting', type=str, default='constant', choices=['constant', 'snr'],
                        help=('Type of loss weighting.  "snr" applies SNR‑based weighting as in the '
                              'Improved DDPM paper.  The default "constant" uses no weighting.'))
    parser.add_argument('--ema_decay', type=float, default=None,
                        help='Decay rate for exponential moving average of model parameters.  Disabled if None.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for the noise predictor network.')
    parser.add_argument('--num_layers', type=int, default=3,
                        help=('Number of hidden layers in the conditional MLP noise predictor. '
                              'The network has this many hidden layers plus an output layer.'))
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Dimensionality of the timestep embedding.')
    parser.add_argument('--nearest_k', type=int, default=5,
                        help='Neighbourhood size for PRDC metrics.')
    parser.add_argument('--save_dir', type=str, default='outputs',
                        help='Directory to save generated samples and logs.')
    parser.add_argument('--reg_strength', type=float, default=0.0,
                        help=('Strength of the regularisation term applied to the predicted noise.  '
                              'If ``--reg_values`` is provided this value is ignored.'))
    parser.add_argument('--reg_type', type=str, default='iso',
                        choices=['iso', 'iso_frob', 'iso_split', 'iso_logeig', 'iso_bures',
                                 'mean_l2', 'var_l2', 'skew', 'kurt', 'var_mi', 'kl', 'mmd_linear', 'mmd_rbf'],
                        help='Type of regularisation to apply to the predicted noise.')
    parser.add_argument('--snr_gamma', type=float, default=5.0,
                        help=('Saturation parameter for SNR weighting.  Larger values make the weighting '
                              'closer to the simple SNR weighting; smaller values impose a stronger cap.'))
    parser.add_argument('--reg_values', type=str, default=None,
                        help=('Comma‑separated list of regularisation strengths.  When provided, the '
                              'training loop will be repeated for each value in the list.'))
    parser.add_argument('--save_every', type=int, default=5000,
                        help='Save a checkpoint every N optimisation steps (0 disables periodic saving).')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID Number.')

    parser.add_argument('--run_suffix', type=str, default='',
                        help=('Optional suffix appended to the regularisation type directory to '
                              'distinguish between multiple runs. For example, specifying '
                              '--run_suffix run_1 writes outputs to outputs/<dataset>/<reg_type>_run_1/reg_<value>.'))

    parser.add_argument('--val_split', type=float, default=0.2,
                        help=('Fraction of the training data to hold out for validation.  '
                              'A non‑zero value enables validation loss tracking and causes '
                              'the best performing model to be saved as ``best.pt`` in each run '
                              'directory.  Set to 0.0 to disable validation.'))
    parser.add_argument('--val_batch_size', type=int, default=None,
                        help=('Batch size for validation.  Defaults to the training batch size '
                              'when not specified.'))

    # Flag to enable or disable extra metrics logging during training.
    # When set, the script computes per‑batch norms and iso values every 100 steps
    # and evaluates PRDC metrics at each checkpoint.  These metrics are written
    # to CSV files in the run directory.  Defaults to False.
    parser.add_argument('--enable_metrics', action='store_true',
                        help='Enable detailed metric logging (norms, iso and PRDC at checkpoints).')

    # When set, the script will generate intermediate samples at all
    # timesteps during the final sampling phase.  The intermediate
    # states of the reverse diffusion process are saved as separate
    # ``.npy`` files in the run directory, one for each timestep.
    # PRDC metrics are computed only on the final timestep samples.
    # This option can dramatically increase memory usage during
    # sampling, so use it judiciously.
    parser.add_argument('--save_intermediate', action='store_true',
                        help=('Generate and save samples at every diffusion '
                              'timestep during final sampling.  PRDC metrics '
                              'are computed on the final timestep only.'))
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Entry point for training one or more diffusion models.

    The function supports training across a list of regularisation strengths via
    the ``--reg_values`` argument.  For each specified value it trains a
    separate model from scratch, saves periodic checkpoints and the final
    generated samples, and reports PRDC metrics.
    """
    # prepare the base output directory organised by dataset and regularisation type
    # Directory structure: save_dir / dataset / reg_type / reg_value
    os.makedirs(args.save_dir, exist_ok=True)
    # ------------------------------------------------------------------
    # Set global random seeds for reproducibility
    #
    # Using a fixed seed across Python's built‑in random module, NumPy and
    # PyTorch reduces variability between runs with the same hyper‑parameters.
    # Note: complete determinism on GPU is not guaranteed; however,
    # enabling deterministic algorithms improves consistency.
    seed = int(args.random_state)
    import random as _random  # import locally to avoid cluttering namespace
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Attempt to enable deterministic operations in PyTorch.  If certain
    # operations lack deterministic implementations, request only a
    # warning rather than raising an error.  For PyTorch versions
    # predating ``use_deterministic_algorithms``, fall back to cuDNN
    # flags.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # Fallback for older PyTorch versions
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # generate the full dataset once; reused across different runs
    # Use a dataset-specific default noise level for moon_circles
    used_noise_level = 0.1 if args.dataset.lower() == 'moon_circles' else args.noise_level
    ds = Synthetic2DDataset(name=args.dataset, num_samples=args.num_samples,
                            noise_level=used_noise_level, random_state=args.random_state)
    X = ds.generate()  # numpy array (num_samples, 2)
    # split into training and validation sets if requested
    val_split = float(args.val_split)
    X_train = X
    X_val = None
    if val_split > 0.0 and 0.0 < val_split < 1.0:
        n = X.shape[0]
        n_val = int(val_split * n)
        # use a NumPy RNG seeded by the same seed for reproducibility
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        val_indices = perm[:n_val]
        train_indices = perm[n_val:]
        X_train = X[train_indices]
        X_val = X[val_indices]
    # convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    # create a generator for deterministic data loading
    gen = torch.Generator()
    gen.manual_seed(seed)
    dataloader = DataLoader(TensorDataset(X_train_tensor), batch_size=args.batch_size,
                             shuffle=True, drop_last=True, generator=gen)
    # create validation DataLoader if applicable
    val_dataloader = None
    if X_val is not None:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        val_batch_size = args.val_batch_size if args.val_batch_size is not None else args.batch_size
        val_dataloader = DataLoader(TensorDataset(X_val_tensor), batch_size=val_batch_size,
                                    shuffle=False)

    # determine list of regularisation strengths
    if args.reg_values:
        try:
            reg_values = [float(v.strip()) for v in args.reg_values.split(',') if v.strip()]
        except ValueError:
            raise ValueError('Invalid format for --reg_values.  Provide a comma‑separated list of floats.')
    else:
        reg_values = [args.reg_strength]
    # derive dataset‑specific and regularisation‑type directories
    dataset_dir = os.path.join(args.save_dir, args.dataset)
    # include run suffix on the reg_type directory if provided
    suffix = f"_{args.run_suffix}" if args.run_suffix else ''
    reg_type_dir = os.path.join(dataset_dir, f"{args.reg_type}{suffix}")
    # create base directories for this dataset and regularisation type (with suffix)
    os.makedirs(reg_type_dir, exist_ok=True)
    # iterate over regularisation strengths
    for reg in reg_values:
        print(f"\nTraining with reg_strength={reg}\n{'-'*40}")
        # set up a subdirectory for this run (dataset/reg_type/reg_value)
        run_dir = os.path.join(reg_type_dir, f'reg_{reg}')
        os.makedirs(run_dir, exist_ok=True)
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        samples_dir = os.path.join(run_dir, "samples")
    
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
    
        # track the best validation loss and corresponding checkpoint for this run
        best_val_loss = float('inf')
        best_ckpt_path = None
        # persist the training dataset for reproducibility
        dataset_filename = f"dataset_{args.dataset}.npy"
        np.save(os.path.join(run_dir, dataset_filename), X)
        # write a summary of the training settings to a text file
        settings_path = os.path.join(run_dir, f'settings_{args.dataset}_reg_{reg}.txt')
        with open(settings_path, 'w') as f:
            f.write('Training settings\n')
            f.write('-' * 40 + '\n')
            f.write(f'dataset_name: {args.dataset}\n')
            f.write(f'num_samples: {args.num_samples}\n')
            f.write(f'noise_level: {used_noise_level}\n')
            f.write(f'random_state: {args.random_state}\n')
            f.write(f'num_diffusion_steps: {args.num_diffusion_steps}\n')
            f.write(f'schedule: {args.schedule}\n')
            f.write(f'train_steps: {args.train_steps}\n')
            f.write(f'batch_size: {args.batch_size}\n')
            f.write(f'learning_rate: {args.lr}\n')
            f.write(f'criterion: {args.criterion}\n')
            f.write(f'weighting: {args.weighting}\n')
            f.write(f'ema_decay: {args.ema_decay}\n')
            f.write(f'hidden_dim: {args.hidden_dim}\n')
            f.write(f'num_layers: {args.num_layers}\n')
            f.write(f'embed_dim: {args.embed_dim}\n')
            f.write(f'nearest_k: {args.nearest_k}\n')
            f.write(f'reg_type: {args.reg_type}\n')
            f.write(f'reg_strength: {reg}\n')
            f.write(f'snr_gamma: {args.snr_gamma}\n')
            f.write(f'save_every: {args.save_every}\n')
            f.write(f'val_split: {args.val_split}\n')
            f.write(f'val_batch_size: {args.val_batch_size}\n')
        # ------------------------------------------------------------------
        # Setup additional metric logging based on a command‑line flag
        #
        # When ``--enable_metrics`` is passed on the command line, the script
        # computes per‑batch norms and iso values every 100 steps and also
        # evaluates PRDC metrics at each checkpoint.  These metrics can be
        # expensive to compute, so they are disabled by default.
        enable_metrics = getattr(args, 'enable_metrics', False)
        # prepare CSV files for incremental metrics
        # Always create a PRDC metrics file at checkpoints
        prdc_ckpt_path = os.path.join(checkpoint_dir,
                                      f'checkpoint_prdc_metrics_{args.dataset}_reg_{reg}.csv')
        with open(prdc_ckpt_path, 'w') as pf:
            pf.write('step,precision,recall,density,coverage\n')
        norm_iso_path = None
        if enable_metrics:
            # create norms/iso file only when extra metrics are enabled
            norm_iso_path = os.path.join(run_dir,
                                         f'norm_iso_metrics_{args.dataset}_reg_{reg}.csv')
            with open(norm_iso_path, 'w') as nf:
                nf.write('step,norm_pred,norm_true,iso\n')
        # build diffusion model
        betas = make_beta_schedule(args.num_diffusion_steps, mode=args.schedule)
        # instantiate the conditional dense model used in the original codebase
        dims = [2] + [args.hidden_dim] * args.num_layers + [2]
        eps_model = ConditionalDenseModel(dims, activation='relu', embed_dim=args.embed_dim)
        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        model = GenericDDPM(eps_model=eps_model, betas=betas,
                            criterion=args.criterion, lr=args.lr,
                            ema_decay=args.ema_decay, device=device,
                            reg_strength=reg, reg_type=args.reg_type,
                            snr_gamma=args.snr_gamma)
        model.to(device)
        # training loop
        data_iter = iter(dataloader)
        for step in range(args.train_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            x0 = batch[0].to(device)
            loss_val = model.train_step(x0, weighting=args.weighting)
            # simple console logging
            # if (step + 1) % 1000 == 0 or step == 0:
            #     print(f'Step {step + 1}/{args.train_steps} - loss: {loss_val:.6f}')
            # ------------------------------------------------------------------
            # Additional metric logging every 100 steps for single runs
            if enable_metrics and ((step + 1) % 100 == 0):
                with torch.no_grad():
                    # sample random timesteps for each example in the batch
                    t_iso = torch.randint(0, model.betas.size(0), (x0.size(0),), device=device)
                    x_t_iso, noise_iso = model.diffuse(x0, t_iso)
                    pred_noise_iso = model.eps_model(x_t_iso, t_iso)
                    # compute average L2 norms of predicted and true noise
                    norm_pred = pred_noise_iso.norm(dim=1).mean().item()
                    norm_true = noise_iso.norm(dim=1).mean().item()
                    # compute iso metric (mean squared norm divided by dimension)
                    dim_pred = pred_noise_iso.shape[-1]
                    iso_val = (pred_noise_iso.pow(2).sum(dim=1).mean().item()) / dim_pred
                # append to the norms/iso CSV
                with open(norm_iso_path, 'a') as nf:
                    nf.write(f"{step + 1},{norm_pred:.6f},{norm_true:.6f},{iso_val:.6f}\n")
            # periodic checkpointing
            if args.save_every and (step + 1) % args.save_every == 0:
                # evaluate validation loss if a validation set exists
                if val_dataloader is not None:
                    # compute validation loss using the model's built‑in evaluation method
                    val_loss = model.evaluate(val_dataloader, weighting=args.weighting)
                    print(f'Validation loss at step {step + 1}: {val_loss:.6f}')
                    # if improved, save as the current best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_ckpt_path = os.path.join(run_dir, 'best.pt')
                        to_save_best = {
                            'model_state_dict': model.state_dict(),
                            'epsilon_model_state_dict': model.eps_model.state_dict(),
                        }
                        if model.ema_model is not None:
                            to_save_best['ema_state_dict'] = model.ema_model.state_dict()
                        torch.save(to_save_best, best_ckpt_path)
                        # print(f'Saved new best model to {best_ckpt_path}')
                # always save the current model as a checkpoint regardless of validation
                ckpt_name = f'checkpoint_step_{step + 1}.pt'
                ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                to_save = {
                    'model_state_dict': model.state_dict(),
                    'epsilon_model_state_dict': model.eps_model.state_dict(),
                }
                if model.ema_model is not None:
                    to_save['ema_state_dict'] = model.ema_model.state_dict()
                torch.save(to_save, ckpt_path)
                # print(f'Saved checkpoint to {ckpt_path}')

                # compute PRDC metrics at this checkpoint
                # sample ``args.num_samples`` points and compute PRDC metrics
                gen_batches_ckpt = []
                remaining_ckpt = args.num_samples
                sample_batch_size = args.batch_size  # reuse training batch size for sampling
                while remaining_ckpt > 0:
                    n_samp = min(sample_batch_size, remaining_ckpt)
                    samples_ckpt = model.sample(n_samp, device=device,
                                                use_ema=args.ema_decay is not None)
                    gen_batches_ckpt.append(samples_ckpt.cpu())
                    remaining_ckpt -= n_samp
                x_gen_ckpt = torch.cat(gen_batches_ckpt, dim=0)
                x_gen_np_ckpt = x_gen_ckpt.numpy().astype(np.float32)
                prdc_ckpt = compute_prdc(real_features=X, fake_features=x_gen_np_ckpt,
                                         nearest_k=args.nearest_k)
                # append to the checkpoint PRDC CSV
                with open(prdc_ckpt_path, 'a') as pf:
                    pf.write(f"{step + 1},{prdc_ckpt['precision']:.6f},{prdc_ckpt['recall']:.6f},"
                             f"{prdc_ckpt['density']:.6f},{prdc_ckpt['coverage']:.6f}\n")
        # sampling after training
        print('Training complete. Sampling in chunks...')
        # if a best validation checkpoint exists, load it before sampling
        if val_dataloader is not None and best_ckpt_path is not None:
            print(f'Loading best model from {best_ckpt_path} for sampling...')
            best_state = torch.load(best_ckpt_path, map_location=device, weights_only=True)
            # load entire model state_dict if available
            if 'model_state_dict' in best_state:
                model.load_state_dict(best_state['model_state_dict'])
            # also load epsilon model explicitly (redundant but safe)
            if 'epsilon_model_state_dict' in best_state:
                model.eps_model.load_state_dict(best_state['epsilon_model_state_dict'])
            # load EMA weights if present
            if model.ema_model is not None and 'ema_state_dict' in best_state:
                model.ema_model.load_state_dict(best_state['ema_state_dict'])
        # ------------------------------------------------------------------
        # Generate samples either at the final timestep only, or at all
        # timesteps if ``--save_intermediate`` is enabled.  Sampling
        # proceeds in chunks to avoid excessive memory usage.  When
        # intermediate steps are saved, the final step is used for
        # PRDC evaluation, and each timestep's samples are saved as
        # separate ``.npy`` files.
        if getattr(args, 'save_intermediate', False):
            # prepare storage for each timestep (T timesteps)
            T = args.num_diffusion_steps
            step_storage: list[list[torch.Tensor]] = [[] for _ in range(T)]
            batch_size = 500  # sample in batches to reduce memory footprint
            remaining = args.num_samples
            while remaining > 0:
                n = min(batch_size, remaining)
                # obtain all intermediate states; shape (T, n, dim)
                all_steps = model.sample(n, device=device,
                                         use_ema=args.ema_decay is not None,
                                         return_all_steps=True)
                # accumulate each timestep into the corresponding list
                for idx in range(all_steps.shape[0]):
                    step_storage[idx].append(all_steps[idx].cpu())
                remaining -= n
            # concatenate across batches for each timestep and save to disk
            final_samples_np = None
            for idx in range(len(step_storage)):
                if step_storage[idx]:
                    cat = torch.cat(step_storage[idx], dim=0)
                else:
                    # should not happen, but guard against empty list
                    cat = torch.empty(0, model.eps_model.layers[-1].out_features if hasattr(model.eps_model, 'layers') else 2)
                arr_np = cat.numpy().astype(np.float32)
                # save the timestep's samples to disk
                step_path = os.path.join(samples_dir, f'samples_step_{idx}.npy')
                np.save(step_path, arr_np)
                # record the final timestep array for PRDC metrics
                if idx == len(step_storage) - 1:
                    final_samples_np = arr_np
            # compute PRDC metrics using only the final timestep samples
            if final_samples_np is None:
                final_samples_np = np.empty((0, X.shape[1]), dtype=np.float32)
            print('Computing PRDC metrics...')
            metrics = compute_prdc(real_features=X, fake_features=final_samples_np,
                                   nearest_k=args.nearest_k)
            for k, v in metrics.items():
                print(f'{k}: {v:.4f}')
            # write PRDC metrics to a CSV file
            metrics_path = os.path.join(run_dir, f'prdc_metrics_{args.dataset}_reg_{reg}.csv')
            with open(metrics_path, 'w') as mf:
                mf.write('metric,value\n')
                for k, v in metrics.items():
                    mf.write(f'{k},{v:.6f}\n')
            # save the final timestep samples separately for convenience
            save_path = os.path.join(run_dir,
                                    f'generated_{args.dataset}_reg_{reg}.npy')
            np.save(save_path, final_samples_np)
            print(f'Saved final generated samples to {save_path}')
            # create a comparison plot using the final timestep samples
            try:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(X[:, 0], X[:, 1], s=2, alpha=0.5, label='Training data')
                ax.scatter(final_samples_np[:, 0], final_samples_np[:, 1], s=2, alpha=0.5, label='Generated samples')
                ax.set_title(f'{args.dataset} vs generated (reg={reg})')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_aspect('equal', adjustable='box')
                ax.legend()
                plot_path = os.path.join(run_dir, f'plot_{args.dataset}_reg_{reg}.png')
                fig.tight_layout()
                fig.savefig(plot_path, dpi=300)
                plt.close(fig)
                print(f'Saved comparison plot to {plot_path}')
            except Exception as e:
                print(f'Failed to create plot: {e}')
        else:
            # default behaviour: generate only final samples in chunks and compute PRDC
            gen_batches = []
            batch_size = 500  # sample in batches to reduce GPU memory footprint
            remaining = args.num_samples
            while remaining > 0:
                n = min(batch_size, remaining)
                batch_samples = model.sample(n, device=device,
                                             use_ema=args.ema_decay is not None)
                gen_batches.append(batch_samples.cpu())
                remaining -= n
            x_gen = torch.cat(gen_batches, dim=0)
            x_gen_np = x_gen.numpy().astype(np.float32)
            # compute PRDC metrics
            print('Computing PRDC metrics...')
            metrics = compute_prdc(real_features=X, fake_features=x_gen_np, nearest_k=args.nearest_k)
            for k, v in metrics.items():
                print(f'{k}: {v:.4f}')
            # write PRDC metrics to a CSV file
            metrics_path = os.path.join(run_dir, f'prdc_metrics_{args.dataset}_reg_{reg}.csv')
            with open(metrics_path, 'w') as mf:
                mf.write('metric,value\n')
                for k, v in metrics.items():
                    mf.write(f'{k},{v:.6f}\n')
            # save generated samples to disk
            save_path = os.path.join(run_dir, f'generated_{args.dataset}_reg_{reg}.npy')
            np.save(save_path, x_gen_np)
            print(f'Saved generated samples to {save_path}')
            # plot the training data and the generated samples side by side
            try:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(X[:, 0], X[:, 1], s=2, alpha=0.5, label='Training data')
                ax.scatter(x_gen_np[:, 0], x_gen_np[:, 1], s=2, alpha=0.5, label='Generated samples')
                ax.set_title(f'{args.dataset} vs generated (reg={reg})')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_aspect('equal', adjustable='box')
                ax.legend()
                plot_path = os.path.join(run_dir, f'plot_{args.dataset}_reg_{reg}.png')
                fig.tight_layout()
                fig.savefig(plot_path, dpi=300)
                plt.close(fig)
                print(f'Saved comparison plot to {plot_path}')
            except Exception as e:
                print(f'Failed to create plot: {e}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
