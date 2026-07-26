"""Single-spin-flip dynamics on a checkerboard-decomposed lattice.

Two acceptance rules live here, and which one you pick is not a matter of
taste when the update is parallel. See ``AcceptanceRule`` below.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from .lattice import checkerboard_mask
from .observables import energy_change_on_flip

EXP_CLIP = 50.0
"""Beyond this the exponential is numerically zero or infinite anyway, and
clipping keeps both backends from emitting overflow warnings."""


class AcceptanceRule(str, Enum):
    """How a proposed flip is accepted.

    ``HEAT_BATH`` is the default, and the reason is subtle enough to be worth
    stating plainly.

    Both rules satisfy detailed balance, so both leave the Boltzmann
    distribution invariant. But invariance is not ergodicity. Metropolis
    accepts ``dE = 0`` with probability *one*, and in a checkerboard update
    every site of the active sublattice is decided simultaneously — so on a
    lattice where zero-energy flips are common, entire sublattices flip in
    deterministic lockstep.

    On the 1D ring this is fatal. A site has two neighbours, so ``dE = 0``
    whenever they are antiparallel, which is most of the time. Configurations
    such as ``(-,-,+,+)`` then form a closed two-state orbit the chain can
    never leave, and a whole family of states becomes unreachable. The sampler
    still returns the Boltzmann distribution — restricted to the subspace it
    can reach, and renormalised, which is exactly what makes the failure hard
    to see by eye.

    Heat-bath (Glauber) accepts ``dE = 0`` with probability one half. The coin
    flip breaks the lockstep and restores irreducibility.
    ``test_metropolis_is_not_ergodic_on_the_1d_ring`` pins this down.
    """

    HEAT_BATH = "heat-bath"
    METROPOLIS = "metropolis"


def acceptance_probability(backend: Any, rule: AcceptanceRule, beta_delta_e: Any) -> Any:
    """Probability of accepting a flip, given ``beta * dE``."""
    xp = backend.xp
    clipped = xp.clip(beta_delta_e, -EXP_CLIP, EXP_CLIP)

    if rule is AcceptanceRule.HEAT_BATH:
        return 1.0 / (1.0 + xp.exp(clipped))

    # Metropolis: min(1, exp(-beta dE)). Clamping the exponent at zero folds
    # the "accept downhill moves unconditionally" branch into one expression.
    return xp.exp(-xp.maximum(clipped, 0.0))


def reshape_for_broadcast(backend: Any, values: Any, ndim: int) -> Any:
    """Shape a per-replica vector so it broadcasts over the lattice axes."""
    return values.reshape((-1,) + (1,) * ndim)


def half_step(
    backend: Any,
    spins: Any,
    shape: tuple[int, ...],
    beta_bcast: Any,
    generator: Any,
    mask: Any,
    coupling: float = 1.0,
    rule: AcceptanceRule = AcceptanceRule.HEAT_BATH,
) -> Any:
    """Propose a flip on every site of one sublattice and accept in parallel.

    This is legitimate because the sites of one colour are conditionally
    independent given the other colour: each site's ``dE`` depends only on
    neighbours that are frozen for this half step.
    """
    xp = backend.xp

    delta_e = energy_change_on_flip(backend, spins, shape, coupling)
    probability = acceptance_probability(backend, rule, beta_bcast * delta_e)
    draw = backend.random_uniform(generator, spins.shape)

    return xp.where((draw < probability) & mask, -spins, spins)


def sweep(
    backend: Any,
    spins: Any,
    shape: tuple[int, ...],
    beta_bcast: Any,
    generator: Any,
    masks: tuple[Any, Any],
    coupling: float = 1.0,
    rule: AcceptanceRule = AcceptanceRule.HEAT_BATH,
) -> Any:
    """One full lattice sweep: both checkerboard colours, in order."""
    for mask in masks:
        spins = half_step(backend, spins, shape, beta_bcast, generator, mask, coupling, rule)
    return spins


def build_masks(backend: Any, shape: tuple[int, ...]) -> tuple[Any, Any]:
    """Both checkerboard colours, built once and reused for the whole run.

    Every linear dimension must be even. A periodic lattice with an odd
    dimension contains odd-length cycles and is therefore **not bipartite** —
    site 0 and site L-1 are neighbours *and* share a colour. Updating them
    together would use stale energies and silently break detailed balance,
    which is far worse than crashing.
    """
    odd = [n for n in shape if n % 2]
    if odd:
        raise ValueError(
            f"every lattice dimension must be even for the checkerboard to be valid; got {shape}. "
            "An odd periodic dimension makes the lattice non-bipartite."
        )
    return (
        checkerboard_mask(backend, shape, 0),
        checkerboard_mask(backend, shape, 1),
    )
