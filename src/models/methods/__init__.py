"""Registry of model builders for different training methods."""

from __future__ import annotations

from importlib import import_module
from typing import Callable

import argparse
import torch

from ..ddpm import GenericDDPM

ModelBuilder = Callable[[argparse.Namespace, torch.device, torch.Tensor, torch.nn.Module, float], GenericDDPM]

_MODEL_MODULES = {
    "ddpm": ".ddpm",
    "snr": ".snr",
    "improved": ".improved",
    "score": ".score",
}


def get_available_model_builders() -> list[str]:
    """Return the list of registered model builders."""
    return sorted(_MODEL_MODULES.keys())


def get_model_builder(name: str) -> ModelBuilder:
    """Load and return the model builder for ``name``."""
    if name not in _MODEL_MODULES:
        available = ", ".join(get_available_model_builders())
        raise KeyError(f"Unknown model builder '{name}'. Available builders: {available}")
    module = import_module(_MODEL_MODULES[name], __name__)
    builder = getattr(module, "build_model", None)
    if builder is None:
        raise AttributeError(f"Model builder module '{name}' does not define 'build_model'.")
    return builder
