"""Lattice geometry: periodic neighbours and checkerboard colouring.

Everything here is dimension-agnostic. The same code drives the 1D chain (which
has an exact solution), the 2D square lattice (Onsager) and the 3D cubic
lattice (the one we actually care about). That is not generality for its own
sake: the lower dimensions are what make the 3D results trustworthy, because
they are the only ones we can check against closed-form answers.

Array layout
------------
Spin arrays have shape ``(n_replicas, *shape)``. Axis 0 is the replica axis —
one independent copy of the system per temperature. That axis is what actually
saturates the GPU: a 20x20x20 lattice is far too small to fill a modern device,
but 40 of them evolving simultaneously is not.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

SPIN_DTYPE = "int8"
"""Spins are +/-1, so int8 is sufficient and keeps four times more lattice in
memory than float32. Neighbour sums reach at most 2*ndim (6 in 3D), which still
fits comfortably in int8."""


class InitialState(str, Enum):
    """How the lattice is seeded before the burn-in.

    ``COLD`` is the default, and on large lattices the choice decides whether
    the results are physics or an artefact.

    A random start is an instantaneous quench from infinite temperature. Below
    T_c the system does not relax smoothly into one of the two ordered states —
    it fragments into domains, and the domain walls then have to coarsen away.
    That coarsening is slow, and it gets slower as the lattice grows: on a 80^3
    lattice a thousand burn-in sweeps leave the system frozen in a multi-domain
    configuration whose energy is already near the ground state while its
    magnetisation is nowhere near saturation.

    Measured on this engine, deep in the ordered phase where the true value is
    0.997:

        L=48, T=2.08   hot start -> 0.2416     cold start -> 0.9929
        L=80, T=1.80   hot start -> 0.5529     cold start -> 0.9972

    The trapped temperature moves around with the lattice size and the seed,
    so the damage shows up as isolated spikes scattered through an otherwise
    healthy curve — which is exactly how it evades a glance at the plot.

    Starting cold sidesteps the quench entirely. Above T_c an ordered start
    disorders within a handful of sweeps, so nothing is lost there.
    """

    COLD = "cold"
    HOT = "hot"


def spatial_axes(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Axes of a ``(n_replicas, *shape)`` array that carry lattice structure."""
    return tuple(range(1, len(shape) + 1))


def random_spins(backend: Any, generator: Any, n_replicas: int, shape: tuple[int, ...]) -> Any:
    """A hot (infinite-temperature) start: every spin independently +/-1."""
    xp = backend.xp
    draw = backend.random_uniform(generator, (n_replicas, *shape))
    return xp.where(draw < 0.5, -1, 1).astype(SPIN_DTYPE)


def aligned_spins(backend: Any, n_replicas: int, shape: tuple[int, ...], value: int = 1) -> Any:
    """A cold start: every spin identical. Used heavily in tests."""
    xp = backend.xp
    return xp.full((n_replicas, *shape), value, dtype=SPIN_DTYPE)


def initial_spins(
    backend: Any,
    generator: Any,
    n_replicas: int,
    shape: tuple[int, ...],
    state: InitialState,
) -> Any:
    """Seed the lattice according to ``state``. See ``InitialState``."""
    if state is InitialState.COLD:
        return aligned_spins(backend, n_replicas, shape)
    return random_spins(backend, generator, n_replicas, shape)


def neighbour_sum(backend: Any, spins: Any, shape: tuple[int, ...]) -> Any:
    """Sum of the 2*ndim nearest neighbours of every site, with periodic
    boundary conditions.

    The boundaries are the whole point. The 2024 implementation used a
    convolution with ``padding='same'``, which pads with **zeros** — sites on
    the surface of the lattice were coupled to phantom neighbours of spin 0
    instead of wrapping around to the opposite face. For a 20^3 lattice almost
    half the sites touch a face, and the resulting surface effects bias exactly
    the critical behaviour the simulation is meant to measure.

    ``roll`` gives periodic boundaries for free, and for a nearest-neighbour
    stencil summing 2*ndim shifted views is also cheaper than a general
    n-dimensional convolution.
    """
    xp = backend.xp
    total = xp.zeros_like(spins)
    for axis in spatial_axes(shape):
        total += xp.roll(spins, 1, axis=axis)
        total += xp.roll(spins, -1, axis=axis)
    return total


def checkerboard_mask(backend: Any, shape: tuple[int, ...], colour: int) -> Any:
    """Boolean mask selecting one sublattice of the bipartite decomposition.

    Metropolis is sequential by construction: accepting a flip changes the
    energy landscape seen by the next spin. On a bipartite lattice that
    dependency can be broken exactly rather than approximately — nearest
    neighbours always differ in the parity of their index sum, so all sites of
    one parity are conditionally independent and can be proposed in parallel
    without altering the Markov chain.

    The 2024 implementation reached for the same idea with stride-2 slicing,
    which is also correct but selects only 1/2**ndim of the lattice (an eighth,
    in 3D). A true two-colour checkerboard updates half the lattice per pass —
    four times more work per kernel launch in 3D, for identical physics.
    """
    xp = backend.xp
    grids = xp.meshgrid(*[xp.arange(n) for n in shape], indexing="ij")
    parity = sum(grids) % 2
    return (parity == colour)[None, ...]
