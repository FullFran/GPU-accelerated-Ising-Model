"""Thermodynamic observables.

Two conventions here differ from the 2024 notebooks, and both change the
numbers you get out. They are documented at the point where they bite.
"""

from __future__ import annotations

from typing import Any

from .lattice import neighbour_sum, spatial_axes


def total_energy(backend: Any, spins: Any, shape: tuple[int, ...], coupling: float = 1.0) -> Any:
    """Total energy ``E = -J * sum_<ij> s_i s_j``, one value per replica.

    Note the factor of one half. ``sum_i s_i * (neighbour sum of i)`` visits
    every bond twice, once from each end. The 2024 implementation summed the
    per-site energies without halving them and therefore reported twice the
    true energy — which propagates as a factor of four into the heat capacity,
    since ``C_v`` depends on the *variance* of E.
    """
    xp = backend.xp
    per_site = spins.astype(xp.float32) * neighbour_sum(backend, spins, shape).astype(xp.float32)
    return -0.5 * coupling * per_site.sum(axis=spatial_axes(shape))


def magnetisation_per_site(backend: Any, spins: Any, shape: tuple[int, ...]) -> Any:
    """Signed magnetisation per site, one value per replica."""
    xp = backend.xp
    return spins.astype(xp.float32).mean(axis=spatial_axes(shape))


def energy_change_on_flip(
    backend: Any, spins: Any, shape: tuple[int, ...], coupling: float = 1.0
) -> Any:
    """``dE`` for flipping each site independently.

    Flipping ``s_i`` maps its local energy ``-J s_i * sum_nb`` to its negative,
    so ``dE = 2 J s_i * sum_nb``. No factor of one half here: this is a local
    quantity, not a sum over bonds.
    """
    xp = backend.xp
    return (
        2.0
        * coupling
        * spins.astype(xp.float32)
        * neighbour_sum(backend, spins, shape).astype(xp.float32)
    )


def susceptibility(backend: Any, abs_m: Any, m_squared: Any, beta: Any, n_sites: int) -> Any:
    """``chi = beta * N * (<m^2> - <|m|>^2)``."""
    return beta * n_sites * (m_squared - abs_m**2)


def heat_capacity(backend: Any, energy: Any, energy_squared: Any, beta: Any, n_sites: int) -> Any:
    """``C_v = beta^2 * (<E^2> - <E>^2) / N``, per site."""
    return beta**2 * (energy_squared - energy**2) / n_sites
