# Architecture

Why this repository is laid out the way it is. The short version: a simulation
that has to run on two very different devices, be validated against exact
mathematics, and produce a reproducible artefact has three distinct concerns,
and keeping them apart is what made the bugs findable.

```
src/ising/
  domain/        lattice geometry, observables, dynamics, exact solutions
  ports/         the array-backend protocol the physics depends on
  adapters/      NumPy (oracle) and CuPy (production)
  application/   the temperature-sweep use case
  infra/         CSV and manifest writers
scripts/         headless entry points: run a sweep, render figures
tests/           validation against ground truth
data/reference/  committed results, so the figures reproduce without a GPU
legacy/          the 2024 notebooks, untouched
```

## The backend port

The simulation core never imports NumPy or CuPy. It receives a `Backend` and
works through it.

This is not architecture for its own sake — it buys three concrete things.

**The same physics code runs on both devices.** NumPy and CuPy expose nearly
identical array APIs, so the port only has to cover what genuinely differs:
random number generation, device transfer, and module identity. Everything in
`domain/` is written once.

**It makes the oracle possible.** The NumPy adapter is not a fallback for
machines without a GPU. It is the reference implementation, and every physics
test runs against it, because on small lattices its output can be compared to
closed-form results and to the full Boltzmann state sum. If CuPy ever disagrees
with NumPy, CuPy is wrong. Without a second independent implementation there is
nothing to cross-check against.

**It closes off a specific performance bug.** The original implementation
allocated random numbers on the host and copied them to the device every step:

```python
torch.rand(dE.shape).to(device)      # allocates on CPU, then transfers
```

For a 100³ lattice that is a host-to-device copy of a million floats per step,
thousands of times over. `random_uniform` is part of the port precisely so that
each backend is required to allocate on its own device, and the mistake cannot
quietly reappear.

The port stays deliberately thin. A protocol that mirrored the whole array API
would be a NumPy reimplementation with extra steps.

## Why the layers are separated

**`domain/` knows physics and nothing else.** No file paths, no CLI, no
plotting, no device management. `exact.py` lives here too — the closed-form
solutions are domain knowledge, not test scaffolding, and they are also what
the CLI uses to centre its default temperature window on `T_c`.

**`application/` is one use case.** `run_sweep` takes a config and a backend and
returns a result object. It decides nothing about where results go.

**`infra/` writes files.** A CSV of the numbers plus a JSON manifest of every
parameter that produced them, seed included.

**`scripts/` are consumers.** `run_sweep.py` draws nothing and opens no windows,
which is what allows it to run unchanged inside a batch job with no display
attached. `plot_sweep.py` reads the committed CSVs and renders figures — it
never runs a simulation.

That last split is the one that matters most in practice. The 2024 version
interleaved simulation and `plt.show()` in the same notebook cells, which meant
the engine could not run anywhere without a display, results could not be
regenerated without re-running the physics, and there was no artefact to
compare two runs against. Separating them turned "the plot looks right" into
"here are the numbers, and here is the manifest that produced them".

## Data flow

```
SweepConfig ─┐
             ├─> run_sweep ──> SweepResult ──> writers ──> CSV + manifest
Backend  ────┘                                                  │
                                                                v
                                                          plot_sweep.py
```

Temperatures are held as **replicas along axis 0** of the spin array. That is
the parallelism that actually matters: a single 20³ lattice is far too small to
occupy a GPU, but forty of them evolving simultaneously is not. It is also why
the sweep is one call rather than a loop over temperatures.

## Measurement strategy

Observables are accumulated **on the device** and reduced in place, with a
single transfer at the end of the run. The 2024 version appended a fresh tensor
every step and measured energy on every sweep even though only the tail was
used. Measuring on a stride after a burn-in keeps one transfer for the whole
run instead of thousands.

Timing calls `deviceSynchronize()` before stopping the clock. CuPy is
asynchronous; timing without it measures how fast Python enqueues kernels, not
how fast they execute.

## Reproducibility

Every run writes a manifest containing the lattice shape, temperatures, sweep
counts, coupling, acceptance rule, initial state and **seed**. Reference results
are committed under `data/reference/`, so the figures in the README can be
regenerated by anyone with no GPU and no Kaggle account.

`test_runs_are_reproducible_from_the_seed` asserts that identical configs
produce identical output.

## Testing strategy

Tests are organised by what they compare against, not by which module they
touch:

- `test_lattice.py` — geometry invariants: periodic neighbour counts,
  translation invariance, and that no two same-colour sites are neighbours
  (the correctness condition for parallel updates)
- `test_observables.py` — conventions: ground-state energy, predicted `ΔE`
  against an actual flip, the bond-counting factor
- `test_physics.py` — ground truth: brute-force state sums, the exact 1D
  solution, Onsager, equilibration

Two tests assert that something **fails**: `test_metropolis_is_not_ergodic_on_
the_1d_ring` and the hot-vs-cold equilibration check. Pinning a known defect is
what keeps a default honest — if a future change ever made either safe, the
test goes red and forces the claim to be re-examined.
