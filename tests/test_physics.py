"""Validation against ground truth.

These are the tests that make the project a physics result rather than a
plot. Each one compares the simulation to something known independently of it:
the exact state sum, the exact 1D solution, and Onsager's exact 2D result.

They are statistical, so tolerances are loose — but they are loose around a
*correct* value, and they fail if the boundary conditions, the energy
convention or the acceptance rule are wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from ising.adapters import NumpyBackend
from ising.application.sweep import SweepConfig, run_sweep
from ising.domain.dynamics import AcceptanceRule
from ising.domain.exact import (
    enumerate_observables,
    ising_1d_energy_per_site,
    onsager_critical_temperature,
    onsager_magnetisation,
)

BACKEND = NumpyBackend()


@pytest.mark.parametrize("shape", [(8,), (4, 4)])
@pytest.mark.parametrize("temperature", [1.5, 2.5, 4.0])
def test_matches_brute_force_state_sum_on_a_tiny_lattice(shape, temperature):
    """The strongest available check: no thermodynamic limit, no approximation.

    On a 4x4 lattice the full 65536-state Boltzmann sum is exact, so any error
    in the sampler shows up directly rather than hiding behind finite-size
    effects. This is the test that caught the non-ergodicity below.
    """
    exact = enumerate_observables(shape, temperature)

    result = run_sweep(
        BACKEND,
        SweepConfig(
            shape=shape,
            temperatures=(temperature,),
            n_sweeps=40_000,
            burn_in=2_000,
            measure_every=2,
            seed=11,
        ),
    )

    assert result.energy_per_site[0] == pytest.approx(exact["energy_per_site"], abs=0.02)
    assert result.abs_magnetisation[0] == pytest.approx(exact["abs_magnetisation"], abs=0.02)


def test_metropolis_is_not_ergodic_on_the_1d_ring():
    """Documents why heat-bath is the default, not a preference.

    Metropolis accepts dE = 0 with probability one. In a checkerboard update
    every site of the active sublattice is decided at once, so on the 1D ring —
    where dE = 0 whenever a site's two neighbours are antiparallel — whole
    sublattices flip in deterministic lockstep and part of the state space
    becomes unreachable.

    The failure is quiet: the chain still samples a Boltzmann distribution,
    just over the wrong support. Only comparison against the exact state sum
    exposes it.
    """
    shape, temperature = (8,), 2.5
    exact = enumerate_observables(shape, temperature)

    def energy_with(rule):
        return run_sweep(
            BACKEND,
            SweepConfig(
                shape=shape,
                temperatures=(temperature,),
                n_sweeps=40_000,
                burn_in=2_000,
                measure_every=2,
                seed=11,
                rule=rule,
            ),
        ).energy_per_site[0]

    heat_bath_error = abs(energy_with(AcceptanceRule.HEAT_BATH) - exact["energy_per_site"])
    metropolis_error = abs(energy_with(AcceptanceRule.METROPOLIS) - exact["energy_per_site"])

    assert heat_bath_error < 0.02, "heat-bath must reproduce the exact state sum"
    assert metropolis_error > 0.05, "the Metropolis bias on the 1D ring is systematic, not noise"


def test_odd_lattices_are_rejected_rather_than_silently_wrong():
    """An odd periodic dimension is not bipartite, so the checkerboard is invalid.

    Site 0 and site L-1 would be neighbours *and* share a colour. Failing loudly
    beats producing plausible numbers from a broken update rule.
    """
    with pytest.raises(ValueError, match="even"):
        run_sweep(
            BACKEND,
            SweepConfig(shape=(5, 5), temperatures=(2.5,), n_sweeps=10, burn_in=1),
        )


def test_reproduces_the_exact_1d_chain_energy():
    """1D Ising is solved exactly at finite size, boundaries included."""
    n_sites = 16
    temperatures = (1.0, 2.0, 3.5)

    result = run_sweep(
        BACKEND,
        SweepConfig(
            shape=(n_sites,),
            temperatures=temperatures,
            n_sweeps=30_000,
            burn_in=2_000,
            measure_every=2,
            seed=5,
        ),
    )

    expected = [ising_1d_energy_per_site(t, n_sites) for t in temperatures]
    np.testing.assert_allclose(result.energy_per_site, expected, atol=0.03)


def test_1d_chain_has_no_phase_transition():
    """The textbook result: order survives only at T = 0 in one dimension.

    A simulation with broken periodic boundaries fails this — an open chain
    orders far more readily than a closed one.
    """
    result = run_sweep(
        BACKEND,
        SweepConfig(
            shape=(64,),
            temperatures=(1.5, 2.5, 4.0),
            n_sweeps=20_000,
            burn_in=2_000,
            measure_every=5,
            seed=9,
        ),
    )

    assert np.all(result.abs_magnetisation < 0.35), "1D must not order at finite temperature"


def test_2d_critical_temperature_matches_onsager():
    """The susceptibility peak must land on Onsager's exact T_c.

    A 16x16 lattice cannot reproduce a true divergence, so the peak is broad
    and shifted by finite-size effects — 12% is the honest tolerance here. What
    it does rule out is a systematically wrong T_c, which is exactly what
    zero-padded boundaries produce.
    """
    t_c = onsager_critical_temperature()
    temperatures = np.linspace(0.6 * t_c, 1.5 * t_c, 30)

    result = run_sweep(
        BACKEND,
        SweepConfig(
            shape=(16, 16),
            temperatures=tuple(float(t) for t in temperatures),
            n_sweeps=6_000,
            burn_in=2_000,
            measure_every=5,
            seed=13,
        ),
    )

    peak = result.temperature[int(np.argmax(result.susceptibility))]
    assert peak == pytest.approx(t_c, rel=0.12)


def test_2d_magnetisation_follows_onsager_deep_in_the_ordered_phase():
    """Well below T_c the finite-size correction is small and Onsager applies."""
    temperatures = (1.2, 1.6, 1.8)

    result = run_sweep(
        BACKEND,
        SweepConfig(
            shape=(24, 24),
            temperatures=temperatures,
            n_sweeps=6_000,
            burn_in=2_000,
            measure_every=5,
            seed=17,
        ),
    )

    expected = [onsager_magnetisation(t) for t in temperatures]
    np.testing.assert_allclose(result.abs_magnetisation, expected, atol=0.05)


def test_runs_are_reproducible_from_the_seed():
    config = SweepConfig(
        shape=(8, 8),
        temperatures=(2.0, 3.0),
        n_sweeps=500,
        burn_in=100,
        measure_every=5,
        seed=42,
    )

    first = run_sweep(BACKEND, config)
    second = run_sweep(BACKEND, config)

    np.testing.assert_array_equal(first.abs_magnetisation, second.abs_magnetisation)
    np.testing.assert_array_equal(first.energy_per_site, second.energy_per_site)


def test_high_temperature_limit_is_disordered():
    """At T >> T_c the system must look like independent coin flips."""
    shape = (32, 32)
    result = run_sweep(
        BACKEND,
        SweepConfig(
            shape=shape,
            temperatures=(50.0,),
            n_sweeps=3_000,
            burn_in=500,
            measure_every=5,
            seed=23,
        ),
    )

    # <|m|> for N independent spins scales as sqrt(2 / (pi N)).
    expected = np.sqrt(2.0 / (np.pi * np.prod(shape)))
    assert result.abs_magnetisation[0] == pytest.approx(expected, abs=0.02)
    assert result.energy_per_site[0] == pytest.approx(0.0, abs=0.05)
