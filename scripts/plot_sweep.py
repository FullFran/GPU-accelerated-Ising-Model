#!/usr/bin/env python3
"""Render the reference results into figures.

Plotting is a separate consumer of the data, never part of the simulation.
That split is what lets the engine run headless in a batch job, and it means
these figures can be regenerated from the committed CSVs by anyone, with no
GPU involved.

    python scripts/plot_sweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display required

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ising.domain.exact import TC_3D  # noqa: E402

DATA = Path("data/reference")
OUT = Path("docs/img")

SIZES = (20, 40, 80)
COLOURS = {20: "#4C72B0", 40: "#DD8452", 80: "#55A868"}


def style() -> None:
    plt.rcParams.update({
        "figure.figsize": (7.0, 4.4),
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "legend.frameon": False,
    })


def load(name: str) -> np.ndarray:
    return np.genfromtxt(DATA / name, delimiter=",", names=True)


def mark_critical(ax) -> None:
    ax.axvline(TC_3D, color="0.35", ls="--", lw=1.2,
               label=f"$T_c = {TC_3D:.3f}$ (reference)")


def plot_magnetisation() -> Path:
    fig, ax = plt.subplots()
    for size in SIZES:
        d = load(f"sweep-3d-L{size}-cupy.csv")
        ax.plot(d["temperature"], d["abs_magnetisation"], "o-", ms=3.5, lw=1.4,
                color=COLOURS[size], label=f"$L = {size}$")
    mark_critical(ax)
    ax.set_xlabel("Temperature $T$  $[J/k_B]$")
    ax.set_ylabel(r"$\langle |m| \rangle$")
    ax.set_title("3D Ising magnetisation — the transition sharpens with lattice size")
    ax.legend()
    path = OUT / "magnetisation.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_susceptibility() -> Path:
    fig, ax = plt.subplots()
    for size in SIZES:
        d = load(f"sweep-3d-L{size}-cupy.csv")
        ax.plot(d["temperature"], d["susceptibility"], "o-", ms=3.5, lw=1.4,
                color=COLOURS[size], label=f"$L = {size}$")
    mark_critical(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Temperature $T$  $[J/k_B]$")
    ax.set_ylabel(r"Susceptibility $\chi$")
    ax.set_title("Susceptibility peaks at the critical temperature")
    ax.legend()
    path = OUT / "susceptibility.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _crossover_sites(runs: list[dict]) -> float:
    """Lattice size where the GPU curve overtakes the CPU curve.

    Both are straight lines in log-log over this range, so interpolate there.
    """
    def curve(backend):
        rows = sorted((r for r in runs if r["backend"] == backend), key=lambda r: r["sites"])
        return (np.log10([r["sites"] for r in rows]),
                np.log10([r["spin_updates_per_second"] for r in rows]))

    cpu_x, cpu_y = curve("numpy")
    gpu_x, gpu_y = curve("cupy")
    grid = np.linspace(cpu_x.min(), cpu_x.max(), 2000)
    gap = np.interp(grid, gpu_x, gpu_y) - np.interp(grid, cpu_x, cpu_y)
    return float(10 ** grid[int(np.argmin(np.abs(gap)))])


def plot_throughput() -> Path:
    runs = json.loads((DATA / "benchmark.json").read_text())["runs"]
    fig, ax = plt.subplots()

    for backend, colour, marker in (("numpy", "#C44E52", "s"), ("cupy", "#55A868", "o")):
        rows = [r for r in runs if r["backend"] == backend]
        ax.plot([r["sites"] for r in rows],
                [r["spin_updates_per_second"] for r in rows],
                marker + "-", color=colour, lw=1.6, ms=6,
                label="CPU (NumPy)" if backend == "numpy" else "GPU (CuPy, P100)")

    # The crossover is the honest part of the story, so it is shaded rather
    # than left for the reader to notice.
    crossover = _crossover_sites(runs)
    ax.axvspan(0, crossover, color="#C44E52", alpha=0.07, lw=0)
    ax.axvline(crossover, color="#C44E52", ls=":", lw=1.2)
    ax.text(crossover * 1.3, 5.5e8,
            f"CPU wins below ~{crossover:,.0f} sites:\ntoo little work to fill a P100",
            ha="left", va="center", fontsize=8.5, color="#8C3B3E")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Lattice sites")
    ax.set_ylabel("Spin updates per second")
    ax.set_title("Throughput — the GPU only wins once there is enough work")
    ax.legend(loc="lower right")
    path = OUT / "throughput.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    for plot in (plot_magnetisation, plot_susceptibility, plot_throughput):
        print(f"wrote {plot()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
