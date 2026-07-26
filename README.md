# GPU-Accelerated Ising Model

Monte Carlo simulation of the Ising model on 1D, 2D and 3D lattices, using
single-spin-flip dynamics with a checkerboard decomposition that makes the
update rule genuinely parallel. It runs on CPU (NumPy) and GPU (CuPy) from the
same physics code.

Heat-bath (Glauber) acceptance is the default; Metropolis-Hastings is available
and [there is a good reason it is not the default](#4-metropolis-is-not-ergodic-under-a-checkerboard-update).

**Every result is validated against ground truth**: the exact 1D solution,
Onsager's exact 2D solution, and brute-force enumeration of the full state
space on small lattices.

---

## This is a remake, and the diff is the point

I wrote the first version of this in February 2024, during my master's in
physics. It was four Jupyter notebooks, no tests, and the plots looked right.

I rebuilt it in 2026 with the architecture I use for production systems. The
rebuild was not cosmetic — **writing the tests surfaced four real physics bugs
that the notebooks had been hiding behind plausible-looking curves.** One of
them was not in the old code at all: it was in the rewrite, and only the exact
state sum caught it.

The 2024 version is preserved untouched in [`legacy/`](legacy/). It is worth
comparing.

### What the tests found

**1. The boundaries were not periodic.**
The original computed nearest-neighbour sums with a convolution using
`padding='same'`, which pads with **zeros**. Sites on the surface of the
lattice were coupled to phantom neighbours of spin 0 instead of wrapping around
to the opposite face. On a 20³ lattice, 49% of sites touch a face. Those
surface effects bias precisely the critical behaviour the simulation exists to
measure.

`test_every_site_has_exactly_two_neighbours_per_axis` fails on the old scheme
and passes on the new one.

**2. The total energy was double-counted.**
`E = -J Σ⟨ij⟩ sᵢsⱼ` sums over *bonds*. Summing per-site energies visits every
bond twice, once from each end, so the missing factor of ½ made the reported
energy exactly twice the true value — and since `C_v` depends on the *variance*
of E, a factor of four in the heat capacity.

**3. Magnetisation was averaged with its sign.**
Below T_c a finite system tunnels between the two ordered states, so ⟨m⟩
averages toward zero even when the system is fully ordered. The observable that
survives finite size is ⟨|m|⟩.

**4. Metropolis is not ergodic under a checkerboard update.**
This one I introduced in the rewrite, and it is the most interesting of the
four.

Metropolis accepts `ΔE = 0` with probability **one**. In a checkerboard update
every site of the active sublattice is decided simultaneously — so wherever
zero-energy flips are common, entire sublattices flip in deterministic
lockstep. On the 1D ring a site has only two neighbours, so `ΔE = 0` whenever
they are antiparallel, which is most of the time. Configurations like
`(−,−,+,+)` then form a closed two-state orbit:

```
(−,−,+,+)  ⇄  (+,+,−,−)      every spin has ΔE = 0, every spin flips, forever
```

The chain can never leave. A whole family of states becomes unreachable.

What makes this genuinely nasty is that **the sampler does not look broken.**
It still returns a Boltzmann distribution — over the wrong support,
renormalised. Sampled probabilities came out at a uniform 1.22× the exact
values, with four states at exactly zero. No curve looks wrong. Only comparison
against the exact 512-state sum exposes it.

The fix is heat-bath (Glauber) dynamics, which accepts `ΔE = 0` with
probability ½. The coin flip breaks the lockstep and restores irreducibility.
Both rules satisfy detailed balance; only one is ergodic here. Metropolis
remains available via `--rule metropolis`, and
`test_metropolis_is_not_ergodic_on_the_1d_ring` asserts that it fails.

**The lesson is the transferable part**: detailed balance is not enough.
Invariance of the target distribution says nothing about whether the chain can
reach all of it, and a non-ergodic sampler fails silently and plausibly.

### And one performance bug

Random numbers were allocated on the host and copied to the device on every
single step:

```python
torch.rand(dE.shape).to(device)      # allocates on CPU, then transfers
```

In a sweep over a 100³ lattice that is a host-to-device copy of a million
floats per step, thousands of times over. The port in `ports/backend.py` exists
partly so this cannot silently happen again — backends must allocate random
numbers on their own device.

---

## Why checkerboard

Metropolis is sequential by construction: accepting a flip changes the energy
landscape the next spin sees. You cannot simply update every site at once.

But a cubic lattice is **bipartite** — nearest neighbours always differ in the
parity of their index sum. So all sites of one parity are conditionally
independent given the other, and can be proposed simultaneously *without
approximating anything*. The Markov chain is unchanged.

The 2024 version reached for the same idea using stride-2 slicing, which is
also correct but selects only 1/2^d of the lattice — an eighth, in 3D. A true
two-colour checkerboard updates half the lattice per pass: **four times more
spins per kernel launch in 3D, for identical physics.**

The second axis of parallelism matters more than it looks: temperatures are
simulated as **replicas along axis 0**. A single 20³ lattice is far too small to
occupy a modern GPU. Forty of them evolving simultaneously is not.

**Every linear dimension must be even.** A periodic lattice with an odd
dimension contains odd-length cycles and is therefore not bipartite — site 0
and site L−1 end up neighbours *and* the same colour. The code raises rather
than accepting it, because a broken update rule still produces smooth,
believable curves.

---

## Usage

```bash
uv venv && uv pip install -e ".[dev,plots]"

# CPU — works anywhere
python scripts/run_sweep.py --dim 2 --size 32 --backend numpy

# GPU
uv pip install -e ".[cuda12]"
python scripts/run_sweep.py --dim 3 --size 100 --backend cupy --sweeps 4000
```

The script writes a CSV and a JSON manifest containing every parameter and the
seed. It draws nothing — plotting is a separate consumer, which is what lets
the same script run headless inside a batch job.

```bash
pytest                      # validate against exact solutions
```

---

## Architecture

```
src/ising/
  domain/        physics: lattice geometry, observables, Metropolis, exact solutions
  ports/         the array-backend protocol the physics depends on
  adapters/      NumPy (reference/oracle) and CuPy (production) implementations
  application/   the temperature-sweep use case
  infra/         CSV and manifest writers
scripts/         headless entry point
tests/           validation against ground truth
legacy/          the 2024 notebooks, untouched
```

The NumPy backend is not a fallback. It is the **oracle**: every physics test
runs against it, because on small lattices its output can be compared to
closed-form results and to the full Boltzmann state sum. If CuPy ever disagrees
with NumPy, CuPy is wrong.

---

## Validation

| Check | Reference |
|---|---|
| Energy and ⟨\|m\|⟩ on 8-site and 4×4 lattices | Brute-force sum over all 2^N states |
| Metropolis fails where heat-bath succeeds | Brute-force sum (ergodicity) |
| Odd lattices are rejected | Bipartiteness of the periodic lattice |
| 1D chain energy per site | Exact finite-size solution |
| 1D has no phase transition | Textbook result |
| 2D critical temperature | Onsager: `T_c = 2/ln(1+√2) ≈ 2.269` |
| 2D magnetisation below T_c | Onsager: `m = [1 − sinh⁻⁴(2βJ)]^(1/8)` |
| High-temperature limit | Independent spins: ⟨\|m\|⟩ → √(2/πN) |
| Ground-state energy | `E = −J·N·d` exactly |
| Predicted ΔE vs actual flip | Internal consistency of the acceptance rule |

3D has no exact solution; the reference value is `β_c ≈ 0.221654626`.

---

## License

MIT
