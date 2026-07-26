# The physics

Background for the simulation in this repository. The [README](../README.md)
covers what was built and what broke; this covers why the model behaves the way
it does and where the numbers we test against come from.

## The model

The Ising model places a spin `sᵢ ∈ {−1, +1}` on every site of a lattice.
Neighbouring spins prefer to align, and temperature fights that preference:

```
E = −J Σ⟨ij⟩ sᵢ sⱼ
```

The sum runs over **bonds**, each counted once. `J > 0` is ferromagnetic. With
`J = 1` and `k_B = 1`, temperature is measured in units of `J/k_B`.

It is the simplest model that undergoes a genuine phase transition, which is
why it remains the standard benchmark for anything that samples equilibrium
statistical mechanics.

### Boundary conditions

We use **periodic** boundaries: the lattice wraps, so site `L−1` neighbours
site `0` along every axis. A finite lattice has a surface, and surface sites
have fewer neighbours and therefore order differently from the bulk. On a 20³
lattice 49% of sites touch a face — the "surface" is nearly the whole system.
Wrapping removes the surface entirely and leaves only the finite-size effects
we can actually reason about.

Periodicity also forces a constraint that is easy to miss: **every linear
dimension must be even.** A periodic lattice with an odd dimension contains
odd-length cycles, so it is not bipartite, and the checkerboard update below
becomes invalid.

## Sampling

### Why single-spin flips need care

The equilibrium distribution is Boltzmann, `P(s) ∝ exp(−βE(s))`. We sample it
with a Markov chain whose stationary distribution is `P`, built from
single-spin flips.

Flipping site `i` maps its local energy `−J sᵢ Σ_nb` to the negative of itself,
so

```
ΔE = 2 J sᵢ Σ_{j ∈ nb(i)} sⱼ
```

Note there is no factor of ½ here: `ΔE` is local, while the total energy sums
over bonds. Conflating the two is a classic way to get an energy that is
exactly twice the truth.

### The checkerboard decomposition

Metropolis is sequential by construction: accepting a flip changes the energy
landscape the next spin sees. That is fatal for a GPU, which wants thousands of
sites updated at once.

The escape is that a cubic lattice is **bipartite**. Colour each site by the
parity of its index sum. Nearest neighbours differ by one in exactly one
coordinate, so they always have opposite parity. Every site of one colour
therefore depends only on sites of the other colour — given one sublattice
frozen, the other's sites are **conditionally independent** and can be updated
simultaneously with no approximation at all. The Markov chain is unchanged.

One full sweep updates both colours in turn.

### Detailed balance is not enough

Two acceptance rules both satisfy detailed balance:

```
Metropolis   A(ΔE) = min(1, e^{−βΔE})
Heat-bath    A(ΔE) = 1 / (1 + e^{βΔE})        (Glauber)
```

Detailed balance guarantees that the Boltzmann distribution is *invariant*. It
guarantees nothing about whether the chain can *reach* all of it.

Metropolis accepts `ΔE = 0` with probability one. Combined with a parallel
sublattice update, every zero-energy site flips **deterministically and
simultaneously**. On the 1D ring a site has two neighbours, so `ΔE = 0`
whenever they are antiparallel — most of the time. The configuration
`(−,−,+,+)` maps to `(+,+,−,−)` and back forever, and a family of states
becomes unreachable.

The chain still samples a Boltzmann distribution — over the wrong support,
renormalised. Heat-bath accepts `ΔE = 0` with probability ½, and the coin flip
restores irreducibility. This repository defaults to heat-bath for that reason;
see `test_metropolis_is_not_ergodic_on_the_1d_ring`.

### Initial conditions are part of the method

A random start is an instantaneous quench from infinite temperature. Below `T_c`
the lattice does not settle into one of the two ordered states — it fragments
into domains, and the domain walls must then coarsen away. Coarsening is slow,
and slows further as the lattice grows.

The diagnostic is a contradiction between observables: energy near the ground
state (near-perfect *local* order) alongside a magnetisation far below
saturation (no *global* alignment). Starting from an aligned lattice avoids the
quench; above `T_c` an ordered start disorders within a few sweeps, so nothing
is lost.

## Observables

| Quantity | Definition | Note |
|---|---|---|
| Energy per site | `E / N` | Ground state: `−J·d` in `d` dimensions |
| Magnetisation | `⟨\|m\|⟩`, `m = (1/N) Σ sᵢ` | Absolute value is required — see below |
| Susceptibility | `χ = β N (⟨m²⟩ − ⟨\|m\|⟩²)` | Diverges at `T_c` in the thermodynamic limit |
| Heat capacity | `C_v = β² (⟨E²⟩ − ⟨E⟩²) / N` | Depends on the *variance* of E |

**Why `⟨|m|⟩` and not `⟨m⟩`.** Below `T_c` an infinite system picks one of the
two ordered states and stays there. A finite one tunnels between them, so `⟨m⟩`
averages toward zero even when the system is perfectly ordered at every instant.
The absolute value is the observable that survives finite size.

## Reference values

These are what the test suite asserts against.

### 1D — exactly solvable, and no phase transition

For a periodic chain of `N` sites, with `K = βJ`:

```
u = −J tanh(K) · [1 + tanh^{N−2}(K)] / [1 + tanh^{N}(K)]
```

The finite-size correction matters at the sizes we simulate. In the
thermodynamic limit this reduces to `u = −J tanh(βJ)`.

There is **no phase transition at finite temperature** in one dimension: a
single domain wall costs finite energy but gains entropy that grows with `N`, so
disorder always wins. `⟨|m|⟩` stays small at any `T > 0`.

### 2D — Onsager

Onsager's exact solution of the square lattice gives

```
T_c = 2J / ln(1 + √2) ≈ 2.269
m   = [1 − sinh⁻⁴(2βJ)]^{1/8}      for T < T_c,  0 otherwise
```

This is the strongest closed-form check available, and the only one that pins
down behaviour at a real critical point.

### 3D — no closed form

There is no exact solution. The reference is the high-precision numerical value

```
β_c ≈ 0.221654626      →      T_c ≈ 4.5115
```

which is what the results in the README are compared against.

### Brute force — the strongest check of all

On a lattice small enough to enumerate every one of its `2^N` configurations,
thermal averages can be computed directly from the definition:

```
⟨A⟩ = Σ_s A(s) e^{−βE(s)} / Σ_s e^{−βE(s)}
```

No thermodynamic limit, no approximation, no appeal to any analytic result. A
4×4 lattice is 65,536 states and runs in under a second. This is what caught
the non-ergodicity above, and no smoother test would have.

## Finite-size effects

A finite lattice cannot produce a true divergence, so `χ` shows a rounded peak
that sharpens and shifts as `L` grows rather than a singularity. Any comparison
of a simulated `T_c` against a thermodynamic-limit value carries that shift as
an irreducible systematic — which is why the tests use tolerances around a
correct value rather than demanding exactness.

The disordered phase gives a free consistency check: at `T ≫ T_c` spins are
essentially independent, so

```
⟨|m|⟩ → √(2 / πN)
```

which falls as `1/√N`. Measured across L = 20 → 40 → 80 the ratios come out at
2.7 and 2.9 against a predicted 2.83.
