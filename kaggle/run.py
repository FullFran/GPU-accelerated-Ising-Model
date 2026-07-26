"""Kaggle batch job: CPU/GPU benchmark and 3D production sweep.

A Kaggle script kernel is a single file, so this one installs the package from
GitHub rather than vendoring a copy that would drift out of sync. Everything
below the install is ordinary library usage — no notebook, no display, no
Drive mount.

Push it with:

    kaggle kernels push -p kaggle
    kaggle kernels status fullfran/ising-gpu-benchmark
    kaggle kernels output fullfran/ising-gpu-benchmark -p results/kaggle
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/FullFran/GPU-accelerated-Ising-Model.git"
BRANCH = "main"
OUTPUT = Path("/kaggle/working")


def install() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"git+{REPO}@{BRANCH}"],
        check=True,
    )


def report_environment() -> dict:
    """Record what we actually ran on. A benchmark without this is a rumour."""
    import cupy as cp

    device = cp.cuda.runtime.getDeviceProperties(0)
    info = {
        "gpu": device["name"].decode(),
        "cupy": cp.__version__,
        "total_memory_gb": round(device["totalGlobalMem"] / 1024**3, 2),
    }
    print(json.dumps(info, indent=2), flush=True)
    return info


def main() -> int:
    install()
    environment = report_environment()

    import numpy as np

    from ising.adapters import get_backend
    from ising.application.sweep import SweepConfig, run_sweep
    from ising.domain.exact import TC_3D
    from ising.infra.writers import write_csv, write_manifest

    temperatures = tuple(float(t) for t in np.linspace(0.4 * TC_3D, 1.6 * TC_3D, 40))
    benchmark = []

    # Same physics, same seed, both backends. The CPU sizes stay small on
    # purpose: the point is the crossover, not a heroic CPU run.
    for backend_name, sizes in (("numpy", (10, 20)), ("cupy", (10, 20, 40, 80))):
        backend = get_backend(backend_name)
        for size in sizes:
            config = SweepConfig(
                shape=(size,) * 3,
                temperatures=temperatures,
                n_sweeps=2_000,
                burn_in=1_000,
                measure_every=10,
            )
            result = run_sweep(backend, config)

            updates = config.n_sites * len(temperatures) * config.n_sweeps
            record = {
                "backend": backend_name,
                "size": size,
                "sites": config.n_sites,
                "seconds": round(result.elapsed_seconds, 3),
                "spin_updates_per_second": updates / result.elapsed_seconds,
            }
            benchmark.append(record)
            print(json.dumps(record), flush=True)

            write_csv(result, OUTPUT / f"sweep-3d-L{size}-{backend_name}.csv")
            write_manifest(result, OUTPUT / f"sweep-3d-L{size}-{backend_name}.json")

    (OUTPUT / "benchmark.json").write_text(
        json.dumps({"environment": environment, "runs": benchmark}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
