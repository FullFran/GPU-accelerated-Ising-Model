"""NumPy backend — the reference implementation.

This is not a fallback. It is the oracle: every physics test runs against it,
because on small lattices we can compare its output to closed-form results and
to brute-force enumeration of the full state space.

If the CuPy backend ever disagrees with this one, the CuPy backend is wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class NumpyBackend:
    name = "numpy"

    @property
    def xp(self) -> Any:
        return np

    def rng(self, seed: int | None) -> np.random.Generator:
        return np.random.default_rng(seed)

    def random_uniform(self, generator: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        return generator.random(shape, dtype=np.float32)

    def to_numpy(self, array: Any) -> np.ndarray:
        return np.asarray(array)

    def is_available(self) -> bool:
        return True
