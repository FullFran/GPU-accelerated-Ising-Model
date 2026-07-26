"""Closed-form and brute-force references.

This module is why the project is testable at all. Monte Carlo output is
stochastic, so "it produced a plausible curve" is not a test. The Ising model
is unusually generous here: in one dimension it is solved exactly, in two
Onsager solved it, and on a small enough lattice the entire state space can be
enumerated. Those three give real ground truth to assert against.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np

# Best-known inverse critical coupling for the 3D cubic lattice. There is no
# closed-form solution in 3D; this is the high-precision Monte Carlo value.
BETA_C_3D = 0.221654626
TC_3D = 1.0 / BETA_C_3D


def onsager_critical_temperature(coupling: float = 1.0) -> float:
    """``T_c = 2J / ln(1 + sqrt(2))`` for the 2D square lattice."""
    return 2.0 * coupling / math.log(1.0 + math.sqrt(2.0))


def onsager_magnetisation(temperature: float, coupling: float = 1.0) -> float:
    """Exact spontaneous magnetisation of the 2D Ising model.

    ``m = [1 - sinh^-4(2 beta J)]^(1/8)`` below T_c, and zero above it.
    """
    if temperature >= onsager_critical_temperature(coupling):
        return 0.0
    beta = 1.0 / temperature
    return (1.0 - math.sinh(2.0 * beta * coupling) ** -4) ** 0.125


def ising_1d_energy_per_site(
    temperature: float, n_sites: int, coupling: float = 1.0
) -> float:
    """Exact energy per site of the periodic 1D chain at finite size.

    ``u = -J tanh(K) [1 + tanh^(N-2)(K)] / [1 + tanh^N(K)]`` with ``K = beta J``.

    The finite-size correction matters: we compare against simulations of
    modest chains, not the thermodynamic limit.
    """
    k = coupling / temperature
    t = math.tanh(k)
    return -coupling * t * (1.0 + t ** (n_sites - 2)) / (1.0 + t**n_sites)


def enumerate_observables(
    shape: tuple[int, ...], temperature: float, coupling: float = 1.0
) -> dict[str, float]:
    """Exact thermal averages by summing over every one of the 2^N states.

    Only tractable for tiny lattices, which is exactly the point: it validates
    the simulation core against the definition of the Boltzmann distribution
    itself, with no approximation and no appeal to the thermodynamic limit.

    Every linear dimension must be at least 3. With a dimension of size 2 the
    periodic wrap makes a site its own neighbour twice over, double-counting
    that bond.
    """
    if any(n < 3 for n in shape):
        raise ValueError("every dimension must be >= 3 for periodic bonds to be distinct")

    n_sites = int(np.prod(shape))
    if n_sites > 20:
        raise ValueError(f"2^{n_sites} states is not tractable; keep the lattice under 20 sites")

    states = np.array(list(product((-1, 1), repeat=n_sites)), dtype=np.int8)
    lattices = states.reshape(-1, *shape)
    lattice_axes = tuple(range(1, len(shape) + 1))

    # Rolling by +1 along one axis pairs each site with its forward neighbour,
    # so every bond is counted exactly once and no factor of one half is needed.
    energy = np.zeros(len(states), dtype=np.float64)
    for axis in lattice_axes:
        energy -= coupling * (lattices * np.roll(lattices, 1, axis=axis)).sum(axis=lattice_axes)

    magnetisation = lattices.mean(axis=lattice_axes)

    beta = 1.0 / temperature
    log_weights = -beta * energy
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()

    return {
        "energy": float((weights * energy).sum()),
        "energy_per_site": float((weights * energy).sum() / n_sites),
        "abs_magnetisation": float((weights * np.abs(magnetisation)).sum()),
        "magnetisation_squared": float((weights * magnetisation**2).sum()),
    }
