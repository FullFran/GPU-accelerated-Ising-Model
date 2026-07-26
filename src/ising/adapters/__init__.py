"""Backend adapters and their selection."""

from __future__ import annotations

from .cupy_backend import CupyBackend, CupyBackendUnavailable
from .numpy_backend import NumpyBackend

__all__ = ["CupyBackend", "CupyBackendUnavailable", "NumpyBackend", "get_backend"]

_BACKENDS = {"numpy": NumpyBackend, "cupy": CupyBackend}


def get_backend(name: str = "auto"):
    """Resolve a backend by name.

    ``auto`` prefers CuPy when a CUDA device is actually present and falls back
    to NumPy otherwise, so the same command works on a laptop and on a GPU box.
    """
    if name == "auto":
        candidate = CupyBackend()
        return candidate if candidate.is_available() else NumpyBackend()

    try:
        return _BACKENDS[name]()
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; choose from auto, numpy, cupy") from None
