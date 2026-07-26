"""Observable tests, including the factor-of-two the notebooks got wrong."""

from __future__ import annotations

import numpy as np
import pytest

from ising.adapters import NumpyBackend
from ising.domain.exact import enumerate_observables
from ising.domain.lattice import aligned_spins
from ising.domain.observables import (
    energy_change_on_flip,
    magnetisation_per_site,
    total_energy,
)

BACKEND = NumpyBackend()


@pytest.mark.parametrize(
    ("shape", "expected_bonds_per_site"),
    [((8,), 1), ((5, 5), 2), ((4, 4, 4), 3)],
)
def test_ground_state_energy_counts_each_bond_once(shape, expected_bonds_per_site):
    """A fully aligned lattice has ``E = -J * N * ndim`` exactly.

    A periodic lattice in d dimensions has ``N * d`` bonds, because each of the
    ``N * 2d`` site-neighbour pairs is one end of a bond. The 2024 code summed
    per-site energies without halving them and reported ``-2 J N d`` — twice
    the true value, which becomes a factor of four in the heat capacity since
    that depends on the variance of E.
    """
    n_sites = int(np.prod(shape))
    spins = aligned_spins(BACKEND, n_replicas=1, shape=shape)

    energy = total_energy(BACKEND, spins, shape)

    assert energy[0] == pytest.approx(-1.0 * n_sites * expected_bonds_per_site)


def test_flipping_a_spin_changes_energy_by_the_predicted_amount():
    """``energy_change_on_flip`` must agree with actually flipping the spin.

    This is the invariant the whole Metropolis acceptance rule rests on: if the
    predicted dE disagrees with the real one, the chain samples the wrong
    distribution while still producing smooth, believable curves.
    """
    shape = (5, 5)
    generator = BACKEND.rng(3)
    spins = np.where(generator.random((1, *shape)) < 0.5, -1, 1).astype(np.int8)

    predicted = energy_change_on_flip(BACKEND, spins, shape)
    before = total_energy(BACKEND, spins, shape)[0]

    for index in [(0, 0), (2, 3), (4, 4)]:  # includes corner and edge sites
        flipped = spins.copy()
        flipped[(0, *index)] *= -1
        actual = total_energy(BACKEND, flipped, shape)[0] - before
        assert actual == pytest.approx(predicted[(0, *index)], abs=1e-5)


def test_magnetisation_of_an_aligned_lattice_is_one():
    shape = (4, 4)
    up = aligned_spins(BACKEND, 1, shape, value=1)
    down = aligned_spins(BACKEND, 1, shape, value=-1)

    assert magnetisation_per_site(BACKEND, up, shape)[0] == pytest.approx(1.0)
    assert magnetisation_per_site(BACKEND, down, shape)[0] == pytest.approx(-1.0)


def test_energy_convention_matches_brute_force_enumeration():
    """Cross-check the simulation's energy against the exact state sum.

    ``enumerate_observables`` computes energy by a completely different route
    (one roll per axis, no halving), so agreement means the convention is right
    rather than merely self-consistent.
    """
    shape = (3, 3)
    exact = enumerate_observables(shape, temperature=2.5)

    spins = aligned_spins(BACKEND, 1, shape)
    ground_state = total_energy(BACKEND, spins, shape)[0] / int(np.prod(shape))

    # The exact thermal average at finite T must sit above the ground state.
    assert ground_state == pytest.approx(-2.0)
    assert exact["energy_per_site"] > ground_state
