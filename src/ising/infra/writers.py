"""Result serialisation.

Results are data, not pictures. The simulation writes a CSV plus a JSON
manifest of the exact parameters that produced it; plotting is a separate
consumer. That split is what makes a run reproducible and what lets the same
engine run headless in a batch job with no display attached.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..application.sweep import SweepResult

COLUMNS = (
    "temperature",
    "abs_magnetisation",
    "magnetisation_squared",
    "energy_per_site",
    "susceptibility",
    "heat_capacity",
)


def write_csv(result: SweepResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack([getattr(result, name) for name in COLUMNS])
    header = ",".join(COLUMNS)
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.10g")
    return path


def write_manifest(result: SweepResult, path: Path) -> Path:
    """Everything needed to reproduce the run, including the seed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    config = asdict(result.config)
    config["temperatures"] = [float(t) for t in config["temperatures"]]
    manifest = {
        "backend": result.backend,
        "elapsed_seconds": result.elapsed_seconds,
        "n_measurements": result.n_measurements,
        "config": config,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
