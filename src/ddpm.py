"""Minimal DDPM/ISO-DDPM implementation for 2D experiments."""

from __future__ import annotations

from typing import Optional, Tuple
import copy

import torch
import torch.nn as nn

from .losses import diffusion_loss


class TimeConditionedMLP(nn.Module):
    """Small MLP with timestep embeddings injected at each layer."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 3, num_steps: int = 1000, embed_dim: int = 64) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        self.num_steps = int(num_steps)
        self.data_dim = int(input_dim)
        self.t_embed = nn.Embedding(self.num_steps, embed_dim)
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [input_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.emb_layers = nn.ModuleList([nn.Linear(embed_dim, dims[i + 1]) for i in range(len(dims) - 1)])
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_embed(t)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h) + self.emb_layers[i](t_emb)
            if i < len(self.layers) - 1:
                h = self.activation(h)
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
