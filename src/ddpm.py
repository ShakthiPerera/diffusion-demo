"""Minimal DDPM/ISO-DDPM implementation for 2D experiments."""

from __future__ import annotations

from typing import Optional, Tuple
import copy

import torch
import torch.nn as nn

from .losses import diffusion_loss


class SinusoidalEncoding(nn.Module):
    """Deterministic sinusoidal timestep encoding."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        embed_dim = int(abs(embed_dim))
        if embed_dim < 2 or embed_dim % 2 != 0:
            raise ValueError("embed_dim must be an even integer >= 2")
        self.embed_dim = embed_dim
        omega = self._make_frequencies()
        self.register_buffer("omega", omega)

    def _make_frequencies(self) -> torch.Tensor:
        i = torch.arange(self.embed_dim // 2).view(1, -1)
        return 1 / (10000 ** (2 * i / self.embed_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.numel() == 1:
            t = t.view(1, 1)
        elif t.ndim == 1:
            t = t.view(-1, 1)
        elif t.ndim != 2 or t.shape[1] != 1:
            raise ValueError(f"Invalid timestep shape: {t.shape}")
        t = t.to(torch.float32)
        batch_size = t.shape[0]
        emb = torch.zeros(batch_size, self.embed_dim, device=t.device, dtype=t.dtype)
        emb[:, 0::2] = torch.sin(self.omega * t)
        emb[:, 1::2] = torch.cos(self.omega * t)
        return emb


def _make_activation(name: str | None) -> nn.Module:
    if name is None or name == "none":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


def _make_dense(in_features: int, out_features: int, activation: str | None) -> nn.Sequential:
    layers = [nn.Linear(in_features, out_features)]
    act = _make_activation(activation)
    layers.append(act)
    return nn.Sequential(*layers)


class LearnableSinusoidalEncoding(nn.Sequential):
    """Sinusoidal encoding followed by small MLP."""

    def __init__(self, num_features: list[int], activation: str = "silu") -> None:
        if len(num_features) < 2:
            raise ValueError("num_features needs at least two entries")
        embed_dim = num_features[0]
        dense_list = []
        for idx, (in_f, out_f) in enumerate(zip(num_features[:-1], num_features[1:])):
            is_not_last = idx < len(num_features) - 2
            dense_list.append(_make_dense(in_f, out_f, activation if is_not_last else None))
        super().__init__(SinusoidalEncoding(embed_dim), *dense_list)


class TimeConditionedDense(nn.Module):
    """Linear layer with learnable sinusoidal timestep conditioning."""

    def __init__(self, in_features: int, out_features: int, activation: str | None, embed_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = _make_activation(activation)
        self.time_embed = LearnableSinusoidalEncoding(
            [embed_dim, out_features, out_features],
            activation=activation if activation is not None else "none",
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = out + self.time_embed(t)
        out = self.activation(out)
        return out


class TimeConditionedMLP(nn.Module):
    """Small MLP with timestep embeddings injected at each layer."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 3, num_steps: int = 1000, embed_dim: int = 64) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        self.num_steps = int(num_steps)
        self.data_dim = int(input_dim)
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [input_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            is_not_last = i < len(dims) - 2
            activation = "silu" if is_not_last else "none"
            self.layers.append(TimeConditionedDense(dims[i], dims[i + 1], activation=activation, embed_dim=embed_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, t)
        return h


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
        data_dim = self.data_dim or (eps_model.layers[-1].out_features if hasattr(eps_model, "layers") else 2)
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
