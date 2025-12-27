# Mini Diffusion (2D)

Lightweight PyTorch demo for experimenting with DDPM and ISO‑regularised DDPM on a few toy 2D datasets. The repo is trimmed to the essentials: a single training script and a handful of supporting modules.

## What’s here
- `train.py` – CLI to train/sample the model and log PRDC metrics/plots.
- `src/datasets.py` – generators for `moon_scatter`, `swiss_roll`, `central_banana`, `moon_circles` (all normalised to `[-1, 1]`).
- `src/ddpm.py` – time‑conditioned MLP and diffusion model with optional EMA and ISO regularisation.
- `src/losses.py` – DDPM loss with min‑SNR weighting and ISO regulariser.
- `src/schedules.py` – beta schedules (cosine/linear/quadratic).
- `src/metrics.py` – PRDC metrics implementation.

Removed: nested `src/*` subpackages, improved/score placeholders, analysis scripts, and shell runners.

## Quickstart
1) Install deps (examples):
   ```bash
   pip install -r requirements.txt
   # or conda env create -f environment.yml
   ```
2) Train and sample (uses CPU or `--gpu_id` if CUDA is available):
   ```bash
   python train.py \
     --dataset moon_scatter \
     --train_steps 5000 \
     --reg_strength 0.0 \
     --weighting constant
   ```
3) Outputs land in `outputs/<dataset>/`:
   - `generated_<dataset>.npy` – sampled points
   - `prdc_<dataset>.txt` – precision/recall/density/coverage
   - `plot_<dataset>.png` – scatter of train vs generated
   - `true_<dataset>.npy` – training data used for the run
   - `settings.txt` – dataset/model/diffusion/training configuration

## Key arguments
- `--dataset` `{moon_scatter, swiss_roll, central_banana, moon_circles}`
- `--num_diffusion_steps` (default 1000), `--schedule` `{cosine, linear, quadratic}`
- `--train_steps`, `--batch_size`, `--lr`
- `--weighting` `{constant, snr}` with `--snr_gamma`
- `--reg_strength` (ISO penalty on predicted noise; set >0 for ISO‑DDPM)
- `--ema_decay` to enable EMA sampling
- `--sample_size` number of generated points after training

## Tips
- Increase `train_steps`/`hidden_dim` for tighter samples; start small for quick tests.
- Use `--reg_strength` (e.g., 0.1–1.0) to encourage isotropy; set to 0 for plain DDPM.
- `--weighting snr` matches the “Improved DDPM” min‑SNR objective; `constant` gives the vanilla loss.
