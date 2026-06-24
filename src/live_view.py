from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

cache_root = Path(__file__).resolve().parents[1] / ".cache"
cache_root.mkdir(parents=True, exist_ok=True)
(cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from .config import ModelConfig, load_bathymetry
from .model import step_forward
from .scenarios import SCENARIOS, Scenario, apply_artificial_barrier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the lake model while it is being computed.")
    parser.add_argument(
        "--scenario",
        choices=[scenario.name for scenario in SCENARIOS],
        default="scenario_1",
        help="Scenario to render.",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of model steps.")
    parser.add_argument("--draw-every", type=int, default=5, help="Refresh the figure every N model steps.")
    parser.add_argument("--dx", type=float, default=1000.0, help="Grid spacing in x direction, meters.")
    parser.add_argument("--dy", type=float, default=1000.0, help="Grid spacing in y direction, meters.")
    parser.add_argument("--dt", type=float, default=5.0, help="Time step in seconds.")
    parser.add_argument("--pause", type=float, default=0.03, help="Pause duration between rendered frames.")
    parser.add_argument("--no-block", action="store_true", help="Exit immediately when the computation finishes.")
    return parser.parse_args()


def find_scenario(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise ValueError(f"Unknown scenario: {name}")


def velocity(U: np.ndarray, V: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    safe_depth = np.where(depth > 0.0, depth, np.nan)
    return U / safe_depth, V / safe_depth


def main() -> None:
    args = parse_args()
    config = replace(ModelConfig(), steps=args.steps, dx=args.dx, dy=args.dy, dt=args.dt)
    scenario = find_scenario(args.scenario)
    original_depth = load_bathymetry(config.bathymetry_path)
    depth = apply_artificial_barrier(original_depth) if scenario.add_barrier else original_depth.copy()
    wet_mask = depth > 0.0

    zeta = np.zeros_like(depth, dtype=float)
    U = np.zeros_like(depth, dtype=float)
    V = np.zeros_like(depth, dtype=float)

    extent = (
        -0.5,
        depth.shape[0] - 0.5,
        -0.5,
        depth.shape[1] - 0.5,
    )
    x_index = np.arange(depth.shape[0])
    y_index = np.arange(depth.shape[1])
    xx, yy = np.meshgrid(x_index, y_index)
    stride = 2

    plt.ion()
    fig, (lake_ax, line_ax) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=120, width_ratios=[1.15, 1.0])
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.11, top=0.88, wspace=0.28)
    zeta_scale = 0.05
    image = lake_ax.imshow(
        np.where(wet_mask, zeta, np.nan).T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="coolwarm",
        vmin=-zeta_scale,
        vmax=zeta_scale,
    )
    lake_ax.contour(x_index, y_index, wet_mask.T.astype(float), levels=[0.5], colors="#333333", linewidths=0.8)
    u, v = velocity(U, V, depth)
    arrows = lake_ax.quiver(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        np.nan_to_num(u.T[::stride, ::stride]),
        np.nan_to_num(v.T[::stride, ::stride]),
        color="black",
        scale=0.35,
        width=0.003,
    )
    lake_ax.set_xlabel("x index")
    lake_ax.set_ylabel("y index")
    colorbar = fig.colorbar(image, ax=lake_ax, shrink=0.82)
    colorbar.set_label("zeta (m)")

    point_i, point_j = config.point_index
    steps: list[int] = []
    point_zeta: list[float] = []
    (line,) = line_ax.plot([], [], color="#1f77b4", linewidth=2.0)
    line_ax.set_title("Sea level at [25, 10]")
    line_ax.set_xlabel("time step")
    line_ax.set_ylabel("zeta (m)")
    line_ax.grid(True, alpha=0.35)

    print(f"Live rendering {scenario.name}: {scenario.description}")
    print(f"dx={config.dx:.1f} m, dy={config.dy:.1f} m, dt={config.dt:.3f} s")

    for step in range(args.steps + 1):
        if step % args.draw_every == 0 or step == args.steps:
            steps.append(step)
            point_zeta.append(float(zeta[point_i, point_j]))

            zeta_scale = max(0.02, float(np.nanmax(np.abs(np.where(wet_mask, zeta, np.nan)))) * 1.1)
            image.set_data(np.where(wet_mask, zeta, np.nan).T)
            image.set_clim(-zeta_scale, zeta_scale)
            u, v = velocity(U, V, depth)
            arrows.set_UVC(
                np.nan_to_num(u.T[::stride, ::stride]),
                np.nan_to_num(v.T[::stride, ::stride]),
            )
            wx, wy = scenario.wind(min(step, args.steps - 1))
            lake_ax.set_title(
                f"{scenario.name} | step {step}/{args.steps} | wind=({wx:.0f}, {wy:.0f}) m/s"
            )

            line.set_data(steps, point_zeta)
            line_ax.relim()
            line_ax.autoscale_view()
            plt.pause(args.pause)

        if step == args.steps:
            break

        wx, wy = scenario.wind(step)
        zeta, U, V = step_forward(zeta, U, V, depth, wx, wy, config)
        zeta = np.where(wet_mask, zeta, 0.0)
        U = np.where(wet_mask, U, 0.0)
        V = np.where(wet_mask, V, 0.0)

    print("Live computation finished.")
    if not args.no_block:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
