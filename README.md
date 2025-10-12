# PyTorch denoising diffusion demo

The repository contains a simple PyTorch-based demonstration of denoising diffusion models.
It just aims at providing a basic understanding of this generative modeling approach.

## Directory layout

```
src/
├── analysis/          # Plotting and evaluation utilities (PRDC, anisotropy, etc.)
├── data/              # Synthetic dataset generators
├── metrics/           # PRDC metrics implementation
├── models/            # Core diffusion models and neural network layers
│   └── methods/       # Model builders per training variant (ddpm, snr, ...)
├── schedules/         # Beta schedule helpers
└── training/
    ├── losses.py      # Shared loss/weighting utilities
    ├── methods/       # Method-specific entry points (ddpm, snr, placeholders)
    └── train.py       # Unified training loop and argument parser

scripts/               # Bash entry points for common experiment sweeps
outputs/               # Default location for generated samples and logs
```

## Usage

Select a training variant via the high-level dispatcher:

```
python main.py --method ddpm --help
```

Available methods can be listed with `python main.py --list-methods`. Current
options:

- `ddpm`: standard constant-weighting DDPM training.
- `snr`: Improved DDPM/SNR-weighted training.
- `improved`, `score`: placeholders ready for future extensions.

The original training module remains accessible for advanced workflows:

```
python train.py --dataset eight_gaussians --train_steps 10000
```

The convenience shell scripts under `scripts/` call the same entry points and
can be used from the repository root, for example:

```
./scripts/run_ddpm_schedule_sweep.sh
```

Existing notebook content is unchanged and still lives under `notebooks/`.
