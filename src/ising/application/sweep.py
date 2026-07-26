"""The temperature sweep use case.

One call simulates every temperature at once. Temperatures are replicas along
axis 0, which is what turns a lattice too small to occupy a GPU into a workload
that does.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..domain import observables as obs
from ..domain.dynamics import AcceptanceRule, build_masks, reshape_for_broadcast, sweep
from ..domain.lattice import InitialState, initial_spins


@dataclass(frozen=True)
class SweepConfig:
    shape: tuple[int, ...]
    temperatures: Sequence[float]
    n_sweeps: int = 2000
    burn_in: int = 1000
    measure_every: int = 10
    coupling: float = 1.0
    seed: int | None = 20240210  # the date of the original commit
    rule: AcceptanceRule = AcceptanceRule.HEAT_BATH
    initial_state: InitialState = InitialState.COLD

    def __post_init__(self) -> None:
        if self.burn_in >= self.n_sweeps:
            raise ValueError("burn_in must be shorter than n_sweeps")
        if self.measure_every < 1:
            raise ValueError("measure_every must be at least 1")

    @property
    def n_sites(self) -> int:
        return int(np.prod(self.shape))


@dataclass(frozen=True)
class SweepResult:
    temperature: np.ndarray
    abs_magnetisation: np.ndarray
    magnetisation_squared: np.ndarray
    energy_per_site: np.ndarray
    susceptibility: np.ndarray
    heat_capacity: np.ndarray
    n_measurements: int
    elapsed_seconds: float
    backend: str
    config: SweepConfig = field(repr=False)


def run_sweep(backend: Any, config: SweepConfig) -> SweepResult:
    xp = backend.xp
    shape = config.shape
    ndim = len(shape)
    n_sites = config.n_sites

    temperatures = np.asarray(config.temperatures, dtype=np.float32)
    beta = xp.asarray(1.0 / temperatures, dtype=xp.float32)
    beta_bcast = reshape_for_broadcast(backend, beta, ndim)

    generator = backend.rng(config.seed)
    spins = initial_spins(backend, generator, len(temperatures), shape, config.initial_state)
    masks = build_masks(backend, shape)

    # Accumulate on the device. The 2024 version appended a fresh observable
    # tensor every single step and measured energy on every sweep even though
    # only the tail was used; measuring on a stride and reducing in place keeps
    # one transfer for the whole run instead of thousands.
    zeros = lambda: xp.zeros(len(temperatures), dtype=xp.float32)  # noqa: E731
    acc_abs_m, acc_m2 = zeros(), zeros()
    acc_e, acc_e2 = zeros(), zeros()
    n_measurements = 0

    start = time.perf_counter()
    for step in range(config.n_sweeps):
        spins = sweep(
            backend, spins, shape, beta_bcast, generator, masks, config.coupling, config.rule
        )

        if step < config.burn_in or (step - config.burn_in) % config.measure_every:
            continue

        magnetisation = obs.magnetisation_per_site(backend, spins, shape)
        energy = obs.total_energy(backend, spins, shape, config.coupling)

        acc_abs_m += xp.abs(magnetisation)
        acc_m2 += magnetisation**2
        acc_e += energy
        acc_e2 += energy**2
        n_measurements += 1

    _synchronise(backend)
    elapsed = time.perf_counter() - start

    mean_abs_m = acc_abs_m / n_measurements
    mean_m2 = acc_m2 / n_measurements
    mean_e = acc_e / n_measurements
    mean_e2 = acc_e2 / n_measurements

    chi = obs.susceptibility(backend, mean_abs_m, mean_m2, beta, n_sites)
    c_v = obs.heat_capacity(backend, mean_e, mean_e2, beta, n_sites)

    return SweepResult(
        temperature=temperatures,
        abs_magnetisation=backend.to_numpy(mean_abs_m),
        magnetisation_squared=backend.to_numpy(mean_m2),
        energy_per_site=backend.to_numpy(mean_e) / n_sites,
        susceptibility=backend.to_numpy(chi),
        heat_capacity=backend.to_numpy(c_v),
        n_measurements=n_measurements,
        elapsed_seconds=elapsed,
        backend=backend.name,
        config=config,
    )


def _synchronise(backend: Any) -> None:
    """Wait for queued GPU work before stopping the clock.

    CuPy calls are asynchronous. Timing without this measures how fast Python
    can enqueue kernels, not how fast they run — a classic way to report a
    speedup that is not there.
    """
    if backend.name != "cupy":
        return
    backend.xp.cuda.runtime.deviceSynchronize()
