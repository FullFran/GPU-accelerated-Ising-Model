"""CuPy backend — the one that does the real work.

CuPy mirrors the NumPy API closely enough that the simulation core is shared
verbatim. The two things that must not be shared are random number generation
and device transfer, which is exactly what the ``Backend`` port isolates.

Importing this module does not require CUDA; instantiating and using it does.
That lets the test suite import everything unconditionally and skip GPU tests
on a machine without a GPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class CupyBackendUnavailable(RuntimeError):
    """Raised when the CuPy backend is used on a machine without CUDA."""


class CupyBackend:
    name = "cupy"

    def __init__(self) -> None:
        self._cp: Any | None = None

    @property
    def xp(self) -> Any:
        if self._cp is None:
            try:
                import cupy as cp
            except ImportError as exc:  # pragma: no cover - depends on host
                raise CupyBackendUnavailable(
                    "CuPy is not installed. Install a build matching your CUDA "
                    "toolkit (e.g. `pip install cupy-cuda12x`), or run with "
                    "`--backend numpy`."
                ) from exc
            self._cp = cp
        return self._cp

    def rng(self, seed: int | None) -> Any:
        return self.xp.random.default_rng(seed)

    def random_uniform(self, generator: Any, shape: tuple[int, ...]) -> Any:
        # Allocated on the device. Never build this on the host and copy it:
        # a per-sweep host-to-device transfer of the full lattice dominates
        # the runtime once the lattice is large.
        return generator.random(shape, dtype=self.xp.float32)

    def to_numpy(self, array: Any) -> np.ndarray:
        return self.xp.asnumpy(array)

    def is_available(self) -> bool:
        try:
            return self.xp.cuda.runtime.getDeviceCount() > 0
        except Exception:  # pragma: no cover - depends on host
            return False
