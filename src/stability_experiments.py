from __future__ import annotations

import csv
import os
import warnings
from dataclasses import replace
from pathlib import Path

cache_root = Path(__file__).resolve().parents[1] / ".cache"
cache_root.mkdir(parents=True, exist_ok=True)
(cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import ModelConfig, load_bathymetry
from .diagnostics import velocity_components
from .model import SimulationResult, run_simulation
from .scenarios import SCENARIOS


PHYSICAL_DURATION_SECONDS = 20_000.0
DT_VALUES_SECONDS = (5, 10, 20, 40, 60, 80, 85, 90, 95, 100, 120)
GRID_SPACING_VALUES_METERS = (100, 150, 200, 225, 250, 300, 500, 750, 1000, 1500, 2000)


def _cfl_numbers(config: ModelConfig, depth: np.ndarray) -> tuple[float, float]:
    wave_speed = float(np.sqrt(config.g * np.nanmax(depth)))
    one_dimensional = wave_speed * config.dt / min(config.dx, config.dy)
    two_dimensional = wave_speed * config.dt * (1.0 / config.dx + 1.0 / config.dy)
    return one_dimensional, two_dimensional


def _summarize_success(result: SimulationResult) -> dict[str, float]:
    zeta = np.where(result.wet_mask, result.zeta, np.nan)
    u, v = velocity_components(result)
    speed = np.sqrt(u * u + v * v)
    mean_zeta_by_frame = np.nanmean(zeta, axis=(1, 2))
    return {
        "max_abs_zeta_m": float(np.nanmax(np.abs(zeta))),
        "max_speed_m_s": float(np.nanmax(speed)),
        "max_abs_mean_zeta_m": float(np.nanmax(np.abs(mean_zeta_by_frame))),
    }


def _run_case(kind: str, value: float, base_config: ModelConfig, depth: np.ndarray) -> dict[str, float | str]:
    if kind == "dt":
        steps = max(1, round(PHYSICAL_DURATION_SECONDS / value))
        config = replace(
            base_config,
            dt=value,
            steps=steps,
            output_every=max(1, steps // 500),
        )
    elif kind == "grid_spacing":
        config = replace(
            base_config,
            dx=value,
            dy=value,
            dt=20.0,
            steps=round(PHYSICAL_DURATION_SECONDS / 20.0),
            output_every=2,
        )
    else:
        raise ValueError(f"Unknown stability experiment kind: {kind}")

    cfl_1d, cfl_2d = _cfl_numbers(config, depth)
    row: dict[str, float | str] = {
        "experiment": kind,
        "parameter_value": value,
        "dx_m": config.dx,
        "dy_m": config.dy,
        "dt_s": config.dt,
        "steps": config.steps,
        "output_every": config.output_every,
        "cfl_1d": cfl_1d,
        "cfl_2d": cfl_2d,
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = run_simulation(SCENARIOS[0], depth, config)
        row.update({"status": "stable", "error": ""})
        row.update(_summarize_success(result))
    except FloatingPointError as exc:
        row.update(
            {
                "status": "unstable",
                "error": str(exc),
                "max_abs_zeta_m": np.nan,
                "max_speed_m_s": np.nan,
                "max_abs_mean_zeta_m": np.nan,
            }
        )
    return row


def _write_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "parameter_value",
        "dx_m",
        "dy_m",
        "dt_s",
        "steps",
        "output_every",
        "cfl_1d",
        "cfl_2d",
        "status",
        "max_abs_zeta_m",
        "max_speed_m_s",
        "max_abs_mean_zeta_m",
        "error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_sweep(rows: list[dict[str, float | str]], x_label: str, output_path: Path) -> None:
    stable = [row for row in rows if row["status"] == "stable"]
    unstable = [row for row in rows if row["status"] == "unstable"]

    x_stable = np.asarray([float(row["parameter_value"]) for row in stable])
    zeta_stable = np.asarray([float(row["max_abs_zeta_m"]) for row in stable])
    speed_stable = np.asarray([float(row["max_speed_m_s"]) for row in stable])
    x_unstable = np.asarray([float(row["parameter_value"]) for row in unstable])

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.0), dpi=180, sharex=True)
    for ax, y_stable, ylabel in (
        (axes[0], zeta_stable, "max |zeta| (m)"),
        (axes[1], speed_stable, "max speed (m/s)"),
    ):
        ax.plot(x_stable, y_stable, marker="o", color="#2F6F95", linewidth=1.8, label="finite run")
        if x_unstable.size:
            y_marker = max(float(np.nanmax(y_stable)) * 1.08, 0.01)
            ax.scatter(x_unstable, np.full_like(x_unstable, y_marker), marker="x", s=70, color="#C43B32", label="failed")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

    axes[1].set_xlabel(x_label)
    fig.suptitle("Scenario 1 numerical stability sweep")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    config = ModelConfig()
    depth = load_bathymetry(config.bathymetry_path)

    rows = []
    for value in DT_VALUES_SECONDS:
        rows.append(_run_case("dt", float(value), config, depth))
    for value in GRID_SPACING_VALUES_METERS:
        rows.append(_run_case("grid_spacing", float(value), config, depth))

    figures_dir = config.output_dir / "figures"
    data_dir = config.output_dir / "data"
    _write_csv(rows, data_dir / "stability_experiments.csv")
    _plot_sweep([row for row in rows if row["experiment"] == "dt"], "dt (s)", figures_dir / "stability_dt_sweep.png")
    _plot_sweep(
        [row for row in rows if row["experiment"] == "grid_spacing"],
        "dx = dy (m)",
        figures_dir / "stability_grid_spacing_sweep.png",
    )


if __name__ == "__main__":
    main()
