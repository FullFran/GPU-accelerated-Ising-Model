"""Geometry tests — including the one that pins down the 2024 boundary bug."""

from __future__ import annotations

import numpy as np
import pytest

from ising.adapters import NumpyBackend
from ising.domain.lattice import (
    aligned_spins,
    checkerboard_mask,
    neighbour_sum,
    random_spins,
)

BACKEND = NumpyBackend()


@pytest.mark.parametrize("shape", [(6,), (5, 5), (4, 4, 4)])
def test_every_site_has_exactly_two_neighbours_per_axis(shape):
    """Periodic boundaries: no site is short-changed, including on the surface.

    This is the regression test for the original implementation, which used a
    convolution with zero padding. Under that scheme the corner of a 3D lattice
    summed 3 neighbours instead of 6, so this assertion fails there while
    passing here.
    """
    spins = aligned_spins(BACKEND, n_replicas=1, shape=shape)
    sums = neighbour_sum(BACKEND, spins, shape)

    expected = 2 * len(shape)
    assert np.all(sums == expected), "sites on the boundary are missing neighbours"


def test_neighbour_sum_is_translation_invariant():
    """A periodic lattice has no special site, so shifting must change nothing."""
    shape = (5, 5)
    generator = BACKEND.rng(7)
    spins = random_spins(BACKEND, generator, 1, shape)

    reference = neighbour_sum(BACKEND, spins, shape)
    shifted = neighbour_sum(BACKEND, np.roll(spins, 2, axis=1), shape)

    np.testing.assert_array_equal(np.roll(reference, 2, axis=1), shifted)


@pytest.mark.parametrize("shape", [(6,), (4, 4), (4, 4, 4)])
def test_checkerboard_colours_partition_the_lattice(shape):
    black = checkerboard_mask(BACKEND, shape, 0)
    white = checkerboard_mask(BACKEND, shape, 1)

    assert np.all(black ^ white), "every site must have exactly one colour"
    assert black.sum() == white.sum(), "a bipartite lattice splits evenly"


@pytest.mark.parametrize("shape", [(6,), (4, 4), (4, 4, 4)])
def test_same_colour_sites_are_never_neighbours(shape):
    """The correctness condition for parallel updates.

    If two sites of the same colour were neighbours, flipping them
    simultaneously would use stale energies and silently break detailed
    balance. Counting same-colour neighbours must give exactly zero.
    """
    black = checkerboard_mask(BACKEND, shape, 0).astype(np.int8)
    same_colour_neighbours = neighbour_sum(BACKEND, black, shape)[black.astype(bool)]

    assert np.all(same_colour_neighbours == 0)


def test_spins_are_only_plus_or_minus_one():
    shape = (4, 4)
    generator = BACKEND.rng(0)
    spins = random_spins(BACKEND, generator, 3, shape)
    assert set(np.unique(spins)).issubset({-1, 1})
