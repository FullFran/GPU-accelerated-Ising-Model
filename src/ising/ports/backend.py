"""Array backend port.

The simulation core never imports NumPy or CuPy directly. It receives a
``Backend`` and works through it, so the exact same physics code runs on CPU
(for tests and small lattices) and on GPU (for production sweeps).

This is deliberately thin. NumPy and CuPy expose nearly identical array APIs,
so the port only has to cover the few places where they genuinely differ:
random number generation, device transfer, and module identity.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Backend(Protocol):
    """Minimal array namespace the simulation core depends on."""

    name: str

    @property
    def xp(self) -> Any:
        """The array module itself (``numpy`` or ``cupy``)."""
        ...

    def rng(self, seed: int | None) -> Any:
        """Return a seeded random generator for this backend."""
        ...

    def random_uniform(self, generator: Any, shape: tuple[int, ...]) -> Any:
        """Uniform floats in [0, 1) allocated *on the target device*.

        Allocating directly on the device matters: creating the array on the
        host and copying it every sweep is one of the dominant costs in a naive
        GPU Metropolis implementation.
        """
        ...

    def to_numpy(self, array: Any) -> np.ndarray:
        """Copy an array back to host memory as a NumPy array."""
        ...

    def is_available(self) -> bool:
        """Whether this backend can actually run here (e.g. CUDA present)."""
        ...
