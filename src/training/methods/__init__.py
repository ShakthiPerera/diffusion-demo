"""Method-specific training entry points."""

from __future__ import annotations

from importlib import import_module
from typing import Callable, Optional, Sequence

MethodRunner = Callable[[Optional[Sequence[str]]], None]

_METHOD_MODULES = {
    "ddpm": ".ddpm",
    "snr": ".snr",
    "improved": ".improved",
    "score": ".score",
}


def get_available_methods() -> list[str]:
    """Return the list of registered training methods."""
    return sorted(_METHOD_MODULES.keys())


def get_method_runner(name: str) -> MethodRunner:
    """Fetch the runner corresponding to ``name``."""
    if name not in _METHOD_MODULES:
        available = ", ".join(get_available_methods())
        raise KeyError(f"Unknown method '{name}'. Available methods: {available}")
    module = import_module(_METHOD_MODULES[name], __name__)
    runner = getattr(module, "run", None)
    if runner is None:
        raise AttributeError(f"Training method '{name}' does not define a 'run' function.")
    return runner
