from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
MPL_CACHE_DIR = ROOT_DIR / ".cache" / "matplotlib"


@dataclass(frozen=True)
class ModelConfig:
    """Numerical and physical parameters for the linear SWE model."""

    # Grid spacing and time step are assignment assumptions, not read from the data file.
    dx: float = 1000.0
    dy: float = 1000.0
    dt: float = 20.0
    steps: int = 1000
    output_every: int = 1
    g: float = 9.81
    friction: float = 0.003
    wind_stress: float = 3.2e-6
    latitude_deg: float = 60.0
    bathymetry_path: Path = DATA_DIR / "bathymetry.txt"
    output_dir: Path = OUTPUT_DIR
    transect_x_index: int = 25
    point_index: tuple[int, int] = (25, 10)

    @property
    def coriolis_f(self) -> float:
        """Return the Coriolis parameter at the configured latitude."""

        omega = 7.2921159e-5
        return 2.0 * omega * np.sin(np.deg2rad(self.latitude_deg))


def load_bathymetry(path: Path | str) -> np.ndarray:
    """Load the lake depth grid, where 0 means land or closed boundary."""

    data = np.loadtxt(path, dtype=float)
    if data.ndim != 2:
        raise ValueError("Bathymetry must be a two-dimensional grid.")
    return data
