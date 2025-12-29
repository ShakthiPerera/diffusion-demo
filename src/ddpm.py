"""Minimal DDPM/ISO-DDPM implementation for 2D experiments."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple
import copy

import torch
import torch.nn as nn

from .losses import diffusion_loss


def make_activation(mode: str | None) -> nn.Module:
    """Create activation matching the original DDPM MLP design."""

    if mode is None or mode == "none":
        activation = nn.Identity()
    elif mode == "sigmoid":
        activation = nn.Sigmoid()
    elif mode == "tanh":
        activation = nn.Tanh()
    elif mode == "relu":
        activation = nn.ReLU()
    elif mode == "leaky_relu":
        activation = nn.LeakyReLU()
    elif mode == "elu":
        activation = nn.ELU()
    elif mode == "softplus":
        activation = nn.Softplus()
    elif mode in ("swish", "silu"):
        activation = nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {mode}")
    return activation


def make_dense(in_features: int, out_features: int, activation: str | None) -> nn.Sequential:
    linear = nn.Linear(in_features, out_features, bias=True)
    activation_layer = make_activation(activation)
    return nn.Sequential(linear, activation_layer)


class SinusoidalEncoding(nn.Module):
    """Sinusoidal position encoding matching the previous DDPM implementation."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()

        embed_dim = int(abs(embed_dim))
        if embed_dim < 2:
            raise ValueError("At least two embedding dimensions required")
        if embed_dim % 2 != 0:
            raise ValueError("Dimensionality has to be an even number")

        self.embed_dim = embed_dim

        omega = self._make_frequencies()
        self.register_buffer("omega", omega)

    def _make_frequencies(self) -> torch.Tensor:
        i = torch.arange(self.embed_dim // 2).view(1, -1)
        omega = 1 / (10000 ** (2 * i / self.embed_dim))
        return omega

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.numel() == 1:
            t = t.view(1, 1)
        elif t.ndim == 1:
            t = t.view(-1, 1)
        elif t.ndim != 2 or t.shape[1] != 1:
            raise ValueError(f"Invalid shape encountered: {t.shape}")

        device = t.device
        t = t.to(torch.float32)
        batch_size = t.shape[0]

        emb = torch.zeros(batch_size, self.embed_dim, device=device)
        emb[:, 0::2] = torch.sin(self.omega * t)
        emb[:, 1::2] = torch.cos(self.omega * t)
        return emb


class LearnableSinusoidalEncoding(nn.Sequential):
    """Learnable sinusoidal position encoding (sinusoidal + dense stack)."""

    def __init__(self, num_features: Sequence[int], activation: str = "relu") -> None:
        if len(num_features) < 2:
            raise ValueError("Number of features needs at least two entries")

        num_dense_layers = len(num_features) - 1

        embed_dim = num_features[0]
        sinusoidal_encoding = SinusoidalEncoding(embed_dim=embed_dim)

        dense_list = []
        for idx, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            is_not_last = idx < num_dense_layers - 1

            dense = make_dense(
                in_features,
                out_features,
                activation=activation if is_not_last else None,
            )

            dense_list.append(dense)

        super().__init__(sinusoidal_encoding, *dense_list)


class ConditionalDense(nn.Module):
    """Fully connected layer with learnable sinusoidal timestep conditioning."""

    def __init__(self, in_features: int, out_features: int, activation: str | None = "relu", embed_dim: int | None = None) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.activation = make_activation(activation)

        if embed_dim is not None:
            self.emb = LearnableSinusoidalEncoding(
                [embed_dim, out_features, out_features],
                activation=activation if activation is not None else "none",
            )
        else:
            self.emb = None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)

        if self.emb is not None:
            emb = self.emb(t)
            out = out + emb

        if self.activation is not None:
            out = self.activation(out)

        return out


class ConditionalDenseModel(nn.Module):
    """Stack of conditional dense layers mirroring the previous DDPM MLP."""

    def __init__(self, num_features: Sequence[int], activation: str = "relu", embed_dim: int | None = None) -> None:
        super().__init__()

        if len(num_features) < 2:
            raise ValueError("Number of features needs at least two entries")

        num_layers = len(num_features) - 1

        dense_list = []
        for idx, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            is_not_last = idx < num_layers - 1

            dense = ConditionalDense(
                in_features,
                out_features,
                activation=activation if is_not_last else None,
                embed_dim=embed_dim,
            )

            dense_list.append(dense)

        self.dense_layers = nn.ModuleList(dense_list)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for dense in self.dense_layers:
            x = dense(x, t)
        return x


class TimeConditionedMLP(nn.Module):
    """Small MLP with timestep embeddings injected at each layer (previous design)."""

    def __init__(self, num_features: Sequence[int] = (2, 128, 128, 128, 2), activation: str = "relu", embed_dim: int = 128, num_steps: int = 1000) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.data_dim = int(num_features[0])
        self.model = ConditionalDenseModel(num_features=num_features, activation=activation, embed_dim=embed_dim)
        self.data_dim = int(num_features[-1])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.view(1, 1)
        elif t.ndim == 1:
            t = t.view(-1, 1)
        elif t.ndim != 2 or t.shape[1] != 1:
            raise ValueError(f"Invalid timestep shape: {t.shape}")

        t = t.to(dtype=x.dtype) + 1.0
        return self.model(x, t)


class DiffusionModel(nn.Module):
    """Unified diffusion model with optional iso regularisation."""

    def __init__(
        self,
        eps_model: nn.Module,
        betas: torch.Tensor,
        lr: float = 1e-4,
        ema_decay: Optional[float] = None,
        device: Optional[torch.device | str] = None,
        reg_strength: float = 0.0,
    ) -> None:
        super().__init__()
        self.eps_model = eps_model
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        betas = betas.to(torch.float32)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.reg_strength = float(reg_strength)
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=lr)
        self.ema_decay = ema_decay
        self.ema_model = self._clone_model() if ema_decay is not None else None
        self.data_dim = getattr(self.eps_model, "data_dim", None)

    # ------------------------------------------------------------------
    def _clone_model(self) -> nn.Module:
        ema = copy.deepcopy(self.eps_model)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    def diffuse(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def train_step(self, x0: torch.Tensor) -> float:
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
            reg_strength=self.reg_strength,
        )
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        if self.ema_model is not None:
            with torch.no_grad():
                for ema_param, param in zip(self.ema_model.parameters(), self.eps_model.parameters()):
                    ema_param.mul_(self.ema_decay).add_(param.detach() * (1.0 - self.ema_decay))
        return float(total_loss.detach().item())

    def sample(self, num_samples: int, device: Optional[torch.device | str] = None, use_ema: bool = False) -> torch.Tensor:
        device = torch.device(device) if device is not None else self.device
        self.eval()
        eps_model = self.ema_model if (use_ema and self.ema_model is not None) else self.eps_model
        data_dim = self.data_dim or 2
        x_t = torch.randn(num_samples, data_dim, device=device)
        T = self.betas.size(0)
        for t in reversed(range(T)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            beta_t = self.betas[t]
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]
            eps_theta = eps_model(x_t, t_batch)
            model_mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alpha_bar_t * eps_theta)
            if t > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(beta_t)
                x_t = model_mean + sigma_t * noise
            else:
                x_t = model_mean
        self.train()
        return x_t.detach()
