#!/usr/bin/env python3
"""Headless entry point: run a temperature sweep and write results to disk.

This is the file Kaggle executes. It draws nothing and opens no windows — the
same script produces the data locally on CPU and on a P100 in a batch job, with
only ``--backend`` and the lattice size changing.

    python scripts/run_sweep.py --size 20 --dim 3 --backend numpy
    python scripts/run_sweep.py --size 100 --dim 3 --backend cupy --sweeps 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ising.adapters import get_backend  # noqa: E402
from ising.application.sweep import SweepConfig, run_sweep  # noqa: E402
from ising.domain.dynamics import AcceptanceRule  # noqa: E402
from ising.domain.exact import TC_3D, onsager_critical_temperature  # noqa: E402
from ising.infra.writers import write_csv, write_manifest  # noqa: E402

KAGGLE_OUTPUT = Path("/kaggle/working")


def default_output_dir() -> Path:
    """Write where the host expects. Kaggle only preserves /kaggle/working."""
    return KAGGLE_OUTPUT if KAGGLE_OUTPUT.is_dir() else Path("results")


def critical_temperature(dim: int) -> float | None:
    return {1: None, 2: onsager_critical_temperature(), 3: TC_3D}.get(dim)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--size", type=int, default=20, help="linear lattice size L (even)")
    parser.add_argument("--dim", type=int, default=3, choices=(1, 2, 3), help="lattice dimension")
    parser.add_argument("--backend", default="auto", choices=("auto", "numpy", "cupy"))
    parser.add_argument("--sweeps", type=int, default=2000, help="total lattice sweeps")
    parser.add_argument("--burn-in", type=int, default=1000, help="sweeps before measuring")
    parser.add_argument("--measure-every", type=int, default=10, help="stride after burn-in")
    parser.add_argument("--temp-min", type=float, default=None)
    parser.add_argument("--temp-max", type=float, default=None)
    parser.add_argument("--temp-points", type=int, default=40, help="temperatures run in parallel")
    parser.add_argument(
        "--rule",
        default=AcceptanceRule.HEAT_BATH.value,
        choices=[rule.value for rule in AcceptanceRule],
        help="acceptance rule; metropolis is not ergodic under a checkerboard update",
    )
    parser.add_argument("--seed", type=int, default=20240210)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", default=None, help="suffix for the output filenames")
    return parser.parse_args(argv)


def temperature_range(args: argparse.Namespace) -> np.ndarray:
    """Default to a window centred on the known critical temperature."""
    t_c = critical_temperature(args.dim)
    if args.temp_min is not None and args.temp_max is not None:
        low, high = args.temp_min, args.temp_max
    elif t_c is None:  # 1D has no phase transition; sweep a plain range
        low, high = 0.2, 5.0
    else:
        low, high = 0.4 * t_c, 1.6 * t_c
    return np.linspace(low, high, args.temp_points, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backend = get_backend(args.backend)
    shape = (args.size,) * args.dim

    config = SweepConfig(
        shape=shape,
        temperatures=tuple(float(t) for t in temperature_range(args)),
        n_sweeps=args.sweeps,
        burn_in=args.burn_in,
        measure_every=args.measure_every,
        seed=args.seed,
        rule=AcceptanceRule(args.rule),
    )

    tag = args.tag or f"{args.dim}d-L{args.size}-{backend.name}"
    print(f"backend={backend.name}  lattice={shape}  sites={config.n_sites:,}", flush=True)
    print(
        f"rule={config.rule.value} sweeps={config.n_sweeps} "
        f"burn_in={config.burn_in} temperatures={len(config.temperatures)}",
        flush=True,
    )

    result = run_sweep(backend, config)

    out_dir = args.output_dir or default_output_dir()
    csv_path = write_csv(result, out_dir / f"sweep-{tag}.csv")
    manifest_path = write_manifest(result, out_dir / f"sweep-{tag}.json")

    updates = config.n_sites * len(config.temperatures) * config.n_sweeps
    sites_per_second = updates / result.elapsed_seconds
    print(f"elapsed={result.elapsed_seconds:.2f}s measurements={result.n_measurements}", flush=True)
    print(f"throughput={sites_per_second:.3e} spin-updates/s", flush=True)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
