"""
A unified Denoising Diffusion Probabilistic Model (DDPM) implementation.

This module defines two classes: :class:`SimpleMLP`, a lightweight neural
network for predicting noise conditioned on a timestep, and
:class:`GenericDDPM`, which wraps a noise predictor and provides
forward/backward diffusion, training, and sampling routines.  The goal is
to avoid the proliferation of nearly identical diffusion model variants
found in the original codebase by exposing optional features (e.g. EMA)
through parameters rather than separate subclasses.

The default network architecture is a modest multilayer perceptron with a
learnable embedding for timesteps.  For many low‑dimensional synthetic
datasets this is sufficient; users may substitute their own models as
long as they implement a ``forward(x, t)`` method.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from ..schedules.schedules import make_beta_schedule
from ..training.losses import diffusion_loss, build_reg_loss_fn


class SimpleMLP(nn.Module):
    """A simple MLP that predicts noise given a noisy sample and a timestep.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the data (e.g. 2 for 2D points).
    hidden_dim : int, optional
        Number of hidden units per layer.  Default is 128.
    num_layers : int, optional
        Total number of linear layers.  Must be at least 2.  Default is 3.
    num_steps : int, optional
        Number of diffusion steps.  This defines the range of allowable
        timestep indices for the embedding.  Default is 1000.
    embed_dim : int, optional
        Dimensionality of the timestep embedding.  Default is 64.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128,
                 num_layers: int = 3, num_steps: int = 1000, embed_dim: int = 64) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError('num_layers must be at least 2')
        self.num_steps = int(num_steps)
        # embedding for timesteps 0..num_steps-1
        self.t_embed = nn.Embedding(self.num_steps, embed_dim)
        # construct linear layers
        dims = [input_dim + embed_dim] + [hidden_dim] * (num_layers - 2) + [input_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)
            Noisy samples at timestep ``t``.
        t : Tensor of shape (batch_size,)
            Integer timesteps in ``[0, num_steps - 1]``.

        Returns
        -------
        Tensor of shape (batch_size, input_dim)
            Predicted noise residuals.
        """
        # embed timesteps and concatenate with inputs
        t_emb = self.t_embed(t)
        h = torch.cat([x, t_emb], dim=1)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h


class ConditionalMLP(nn.Module):
    """Time‑conditioned multilayer perceptron for noise prediction.

    This network mirrors the conditional dense architecture used in the
    original codebase.  It injects a learnable embedding of the timestep
    into every layer by adding a linear projection of the embedding to
    the output of each linear layer.  The number of hidden layers and
    their width are configurable.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data (e.g. 2 for 2D points).
    hidden_dim : int
        Number of units in each hidden layer.
    num_hidden_layers : int
        Number of hidden layers before the final output layer.
    num_steps : int
        Number of diffusion steps (timestep embeddings range from 0 to
        num_steps-1).
    embed_dim : int
        Dimensionality of the timestep embedding.
    activation : callable, optional
        Activation function applied after each hidden layer.  Defaults to
        ``nn.SiLU()``.

    Notes
    -----
    The architecture is: ``[input_dim] -> hidden_dim x num_hidden_layers -> [input_dim]``.
    For each linear layer, an additional linear mapping of the timestep
    embedding to the layer's output dimension is added before the
    nonlinearity.  The final layer produces a vector of the same
    dimensionality as the input.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128,
                 num_hidden_layers: int = 3, num_steps: int = 1000,
                 embed_dim: int = 12, activation: Optional[nn.Module] = None) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError('num_hidden_layers must be at least 1')
        self.num_steps = int(num_steps)
        # embedding for timesteps 0..num_steps-1
        self.t_embed = nn.Embedding(self.num_steps, embed_dim)
        self.activation = activation if activation is not None else nn.SiLU()
        # construct layer dimensions: input_dim -> hidden_dim x (num_hidden_layers) -> output_dim
        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [input_dim]
        self.layers = nn.ModuleList()
        self.emb_layers = nn.ModuleList()
        # for each layer, create a linear layer for the inputs and one for the embedding
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.emb_layers.append(nn.Linear(embed_dim, out_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the conditional MLP.

        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)
            Noisy samples at timestep ``t``.
        t : Tensor of shape (batch_size,)
            Integer timesteps in ``[0, num_steps - 1]``.

        Returns
        -------
        Tensor of shape (batch_size, input_dim)
            Predicted noise residuals.
        """
        # embed timesteps once per batch
        t_emb = self.t_embed(t)
        h = x
        for i in range(len(self.layers)):
            # linear transform of the data
            h_lin = self.layers[i](h)
            # linear transform of the timestep embedding
            t_lin = self.emb_layers[i](t_emb)
            h = h_lin + t_lin
            # apply activation except on the last layer
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h


class GenericDDPM(nn.Module):
    """Unified diffusion model with optional EMA and simple training loop.

    The model encapsulates the forward and reverse diffusion processes and
    exposes methods to perform a single training step and to sample new
    data.  A single class suffices to cover common variations such as
    exponential moving average (EMA) parameter tracking and SNR weighting.
    """

    def __init__(self, eps_model: nn.Module, betas: torch.Tensor,
                 criterion: str = 'mse', lr: float = 1e-4,
                 ema_decay: Optional[float] = None,
                 device: Optional[torch.device | str] = None,
                 reg_strength: float = 0.0,
                 reg_type: str = 'iso',
                 snr_gamma: float = 5.0) -> None:
        super().__init__()
        self.eps_model = eps_model
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        # register betas and derived quantities as buffers (non‑trainable)
        betas = betas.to(torch.float32)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # store useful constants
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        # choose loss function
        if criterion == 'mse':
            self.loss_fn = nn.MSELoss()
        elif criterion == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown criterion '{criterion}'.")
        # regularisation parameters
        self.reg_strength = float(reg_strength)
        self.reg_type = reg_type
        self.reg_loss_fn = build_reg_loss_fn(self.reg_type, self.reg_strength, self.loss_fn)
        # gamma parameter controlling SNR weighting saturation
        self.snr_gamma = float(snr_gamma)
        # optimiser
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=lr)
        # optional EMA
        self.ema_decay = ema_decay
        if ema_decay is not None:
            self.ema_model = copy.deepcopy(self.eps_model)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
        else:
            self.ema_model = None

    # ---------------------------------------------------------------------
    def diffuse(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the forward diffusion q(x_t | x_0) for a batch of samples.

        Parameters
        ----------
        x0 : Tensor of shape (batch_size, input_dim)
            Original data samples, assumed to lie in ``[-1, 1]``.
        t : Tensor of shape (batch_size,)
            Integer timesteps in ``[0, T-1]``.
        noise : Tensor of shape (batch_size, input_dim), optional
            Optional Gaussian noise.  If ``None`` a new noise tensor is
            sampled from ``N(0, I)``.

        Returns
        -------
        x_t : Tensor
            Noisy samples at time ``t``.
        noise : Tensor
            The noise that was added.  Returned so that the caller can reuse
            it when computing the loss.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # gather scaling factors for each sample in the batch
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def train_step(self, x0: torch.Tensor, weighting: str = 'constant') -> float:
        """Perform a single optimisation step on a batch of data.

        The input ``x0`` should contain uncorrupted data points.  Random
        timesteps are sampled uniformly for each sample in the batch.  The
        model predicts the added noise and the loss is computed between the
        predicted and true noise.  If ``weighting`` is set to ``'snr'`` the
        loss is weighted by the signal‑to‑noise ratio (SNR) as proposed in
        *Improved DDPMs*.

        Parameters
        ----------
        x0 : Tensor of shape (batch_size, input_dim)
            Real data samples.
        weighting : {'constant', 'snr'}, optional
            Type of loss weighting.  Default is ``'constant'``.

        Returns
        -------
        float
            The detached loss value for logging.
        """
        batch_size = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.betas.size(0), (batch_size,), device=device)
        x_t, noise = self.diffuse(x0, t)
        pred_noise = self.eps_model(x_t, t)
        alpha_bar = self.alphas_cumprod[t]
        total_loss, _ = diffusion_loss(
            pred_noise=pred_noise,
            true_noise=noise,
            alphas_cumprod_t=alpha_bar,
            weighting=weighting,
            snr_gamma=self.snr_gamma,
            reg_type=self.reg_type,
            reg_loss_fn=self.reg_loss_fn,
        )
        # optimisation step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        # update EMA parameters
        if self.ema_model is not None:
            with torch.no_grad():
                for ema_param, param in zip(self.ema_model.parameters(), self.eps_model.parameters()):
                    ema_param.mul_(self.ema_decay).add_(param.detach() * (1.0 - self.ema_decay))
        return float(total_loss.detach().item())

    # ------------------------------------------------------------------
    def val_step(self, x0: torch.Tensor, weighting: str = 'constant') -> float:
        """Compute the diffusion loss on a batch without updating parameters.

        This mirrors the behaviour of :meth:`train_step` but omits the
        optimisation steps.  It computes the mean squared error between
        the predicted noise and the true noise with optional SNR
        weighting and regularisation.  The returned loss is detached
        from the computational graph.

        Parameters
        ----------
        x0 : Tensor of shape (batch_size, input_dim)
            Uncorrupted data samples.
        weighting : {'constant', 'snr'}, optional
            Type of loss weighting.  Default is ``'constant'``.

        Returns
        -------
        float
            The detached loss value for logging or validation.
        """
        batch_size = x0.size(0)
        device = x0.device
        # sample random timesteps for each sample in the batch
        t = torch.randint(0, self.betas.size(0), (batch_size,), device=device)
        with torch.no_grad():
            x_t, noise = self.diffuse(x0, t)
            pred_noise = self.eps_model(x_t, t)
            alpha_bar = self.alphas_cumprod[t]
            total_loss, _ = diffusion_loss(
                pred_noise=pred_noise,
                true_noise=noise,
                alphas_cumprod_t=alpha_bar,
                weighting=weighting,
                snr_gamma=self.snr_gamma,
                reg_type=self.reg_type,
                reg_loss_fn=self.reg_loss_fn,
            )
        return float(total_loss.detach().item())

    def evaluate(self, dataloader, weighting: str = 'constant') -> float:
        """Compute the average loss over all batches in a validation dataset.

        This method iterates over the provided dataloader, computes the
        validation loss on each batch via :meth:`val_step` and returns the
        overall average.  Gradients are not accumulated and the model
        temporarily enters evaluation mode during the loop.

        Parameters
        ----------
        dataloader : DataLoader
            Iterable of validation batches (each a tensor of shape
            ``(batch_size, input_dim)`` or tuple with the first element
            being the data tensor).
        weighting : {'constant', 'snr'}, optional
            Weighting scheme to use when computing the loss.  Default is
            ``'constant'``.

        Returns
        -------
        float
            The average validation loss across all samples in
            ``dataloader``.
        """
        # if no validation loader is provided return +inf so that it never
        # improves the best loss
        if dataloader is None:
            return float('inf')
        was_training = self.training
        self.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                # handle case where dataloader yields (data,) tuples
                if isinstance(batch, (list, tuple)):
                    x0 = batch[0]
                else:
                    x0 = batch
                x0 = x0.to(self.device)
                bsz = x0.size(0)
                loss = self.val_step(x0, weighting=weighting)
                total += loss * bsz
                count += bsz
        # restore training mode
        if was_training:
            self.train()
        return total / max(count, 1)


    def sample(self, num_samples: int, device: Optional[torch.device | str] = None,
               use_ema: bool = False, progress: bool = False,
               return_all_steps: bool = False) -> torch.Tensor:
        """Generate new samples via reverse diffusion.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        device : torch.device or str, optional
            Device on which to perform sampling.  Defaults to the model’s
            device.
        use_ema : bool, optional
            Whether to use the EMA version of the predictor network.  Default
            is ``False``.
        progress : bool, optional
            If ``True``, prints simple progress information.

        Parameters
        ----------
        return_all_steps : bool, optional
            If ``True``, returns a tensor containing the intermediate states
            of the reverse diffusion process at each timestep, with shape
            (T, num_samples, input_dim).  The final sample at timestep
            ``t=0`` is then given by the last slice along the first axis.
            If ``False`` (default), only the final samples at ``t=0`` are
            returned.

        Returns
        -------
        Tensor
            If ``return_all_steps`` is ``False``, a tensor of shape
            (``num_samples``, input_dim) containing samples in the data
            space.  If ``return_all_steps`` is ``True``, a tensor of shape
            (``num_steps``, ``num_samples``, input_dim) containing the
            intermediate states at each timestep (from ``t=T-1`` down to
            ``t=0``), where the last element along the first dimension is
            the final sample.
        """
        device = torch.device(device) if device is not None else self.device
        self.eval()
        # choose the appropriate noise predictor (EMA or current model)
        eps_model = self.ema_model if (use_ema and self.ema_model is not None) else self.eps_model
        T = self.betas.size(0)
        # ------------------------------------------------------------------
        # Determine output dimensionality of the noise predictor.
        #
        # Different models expose their layer structure differently.  For
        # ``SimpleMLP`` the final linear layer resides in ``layers[-1]`` and
        # exposes an ``out_features`` attribute.  For a conditional dense
        # model (see ``dense_layers.ConditionalDenseModel``) the layers are
        # stored in ``dense_layers`` and the underlying linear transform is
        # ``layer.linear``.  As a fallback we perform a dummy forward pass
        # with a zero input to infer the output dimension.
        data_dim: Optional[int] = None
        # Case 1: SimpleMLP or any model exposing ``layers``
        if hasattr(eps_model, 'layers') and isinstance(getattr(eps_model, 'layers'), nn.ModuleList):
            try:
                data_dim = eps_model.layers[-1].out_features
            except Exception:
                data_dim = None
        # Case 2: ConditionalDenseModel or any model exposing ``dense_layers``
        if data_dim is None and hasattr(eps_model, 'dense_layers') and isinstance(getattr(eps_model, 'dense_layers'), nn.ModuleList):
            try:
                last_dense = eps_model.dense_layers[-1]
                # ConditionalDense layers expose a ``linear`` attribute
                if hasattr(last_dense, 'linear'):
                    data_dim = last_dense.linear.out_features
            except Exception:
                data_dim = None
        # Case 3: Fallback dummy forward pass
        if data_dim is None:
            # attempt to infer input dimensionality from the first layer
            in_features = None
            if hasattr(eps_model, 'layers') and isinstance(getattr(eps_model, 'layers'), nn.ModuleList):
                try:
                    in_features = eps_model.layers[0].in_features
                except Exception:
                    in_features = None
            if in_features is None and hasattr(eps_model, 'dense_layers') and isinstance(getattr(eps_model, 'dense_layers'), nn.ModuleList):
                try:
                    last_dense = eps_model.dense_layers[0]
                    if hasattr(last_dense, 'linear'):
                        in_features = last_dense.linear.in_features
                except Exception:
                    in_features = None
            # If unable to infer ``in_features`` then default to 1
            if in_features is None:
                in_features = 1
            # Create a dummy zero input and timestep on the target device
            dummy_x = torch.zeros(1, in_features, device=device)
            dummy_t = torch.zeros(1, dtype=torch.long, device=device)
            try:
                with torch.no_grad():
                    dummy_out = eps_model(dummy_x, dummy_t)
                data_dim = dummy_out.shape[1]
            except Exception:
                # As a last resort default to single dimensional output
                data_dim = 1
        # ------------------------------------------------------------------
        # Initialise the starting noise ``x_T`` with Gaussian noise of shape
        # (num_samples, data_dim)
        x_t = torch.randn(num_samples, data_dim, device=device)
        # If returning intermediate steps, prepare a list to accumulate them.
        if return_all_steps:
            # collect x_t after each reverse step. the total number of recorded steps equals T.
            steps_list = []
        for t in reversed(range(T)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            beta_t = self.betas[t]
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]
            # predict noise
            eps_theta = eps_model(x_t, t_batch)
            # compute the mean of the posterior q(x_{t-1} | x_t, x0)
            model_mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alpha_bar_t * eps_theta)
            if t > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(beta_t)
                x_t = model_mean + sigma_t * noise
            else:
                x_t = model_mean
            # record the sample for this timestep
            if return_all_steps:
                steps_list.append(x_t.detach().clone())
            if progress and (t % max(1, T // 10) == 0):
                print(f'Sampling t={t}')
        self.train()
        # If returning all steps, stack them so that the first dimension corresponds to timesteps.
        if return_all_steps:
            # steps_list is in order of t values from T-1 down to 0; stack as (T, num_samples, data_dim)
            return torch.stack(steps_list, dim=0)
        return x_t.detach()
