from __future__ import annotations

import os
import textwrap
from pathlib import Path

cache_root = Path(__file__).resolve().parents[1] / ".cache"
cache_root.mkdir(parents=True, exist_ok=True)
(cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib

# The non-interactive backend keeps batch figure generation stable on headless systems.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

from .diagnostics import (
    eddy_kinetic_energy,
    masked_stats,
    net_transport_through_channel,
    velocity_components,
    vorticity,
)
from .model import SimulationResult


def _extent(result: SimulationResult) -> tuple[float, float, float, float]:
    nx, ny = result.depth.shape
    return (-0.5, nx - 0.5, -0.5, ny - 0.5)


def _imshow(ax: plt.Axes, field: np.ndarray, result: SimulationResult, title: str, cmap: str = "viridis") -> None:
    # Model arrays are stored as (x, y); imshow expects display rows/columns, so transpose for plotting.
    im = ax.imshow(
        field.T,
        origin="lower",
        extent=_extent(result),
        aspect="equal",
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    plt.colorbar(im, ax=ax, shrink=0.82)


def _lake_axes(result: SimulationResult) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = result.depth.shape
    x_index = np.arange(nx)
    y_index = np.arange(ny)
    return x_index, y_index


def _depth_axes(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = depth.shape
    x_index = np.arange(nx)
    y_index = np.arange(ny)
    return x_index, y_index


def _wind_label(result: SimulationResult, frame: int) -> str:
    wx, wy = result.winds[frame]
    return f"wind=({wx:.0f}, {wy:.0f}) m/s"


def _plot_stream_snapshot(ax: plt.Axes, result: SimulationResult, frame: int, title: str) -> None:
    x_index, y_index = _lake_axes(result)
    u, v = velocity_components(result)
    speed = np.sqrt(u[frame] ** 2 + v[frame] ** 2)
    wet = result.wet_mask
    # Mask land cells so streamlines stop at the lake boundary instead of crossing dry cells.
    u_frame = np.ma.array(u[frame].T, mask=~wet.T)
    v_frame = np.ma.array(v[frame].T, mask=~wet.T)
    color_field = np.ma.array(speed.T, mask=~wet.T)

    ax.contourf(x_index, y_index, wet.T.astype(float), levels=[0.5, 1.5], colors=["#f7fbff"], alpha=1.0)
    ax.contour(x_index, y_index, wet.T.astype(float), levels=[0.5], colors="#888888", linewidths=0.8)
    stream = ax.streamplot(
        x_index,
        y_index,
        u_frame,
        v_frame,
        color=color_field,
        cmap="Spectral_r",
        density=0.95,
        linewidth=0.85,
        arrowsize=0.8,
    )
    ax.set_title(title)
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    ax.set_xlim(x_index.min(), x_index.max())
    ax.set_ylim(y_index.min(), y_index.max())
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(stream.lines, ax=ax, shrink=0.82, label="velocity (m/s)")


def plot_point_timeseries(results: list[SimulationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    for result in results:
        i, j = result.config.point_index
        ax.plot(result.steps, result.zeta[:, i, j], label=result.name)
    ax.set_title("Sea level zeta at grid coordinate [25, 10]")
    ax.set_xlabel("time step")
    ax.set_ylabel("zeta (m)")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_input_maps(depth: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_index, y_index = _depth_axes(depth)
    wet = depth > 0.0

    fig, ax = plt.subplots(figsize=(8, 4.0), dpi=180)
    ax.contourf(x_index, y_index, wet.T.astype(float), levels=[0.5, 1.5], colors=["#f7fbff"], alpha=1.0)
    ax.contour(x_index, y_index, wet.T.astype(float), levels=[0.5], colors="#888888", linewidths=1.0)
    ax.set_title("Original lake geometry")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    ax.set_xlim(x_index.min(), x_index.max())
    ax.set_ylim(y_index.min(), y_index.max())
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_dir / "lake_geometry.png")
    plt.close(fig)

    masked_depth = np.where(wet, depth, np.nan)
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=180)
    im = ax.imshow(
        masked_depth.T,
        origin="lower",
        extent=(-0.5, depth.shape[0] - 0.5, -0.5, depth.shape[1] - 0.5),
        aspect="equal",
        cmap="viridis_r",
    )
    ax.contour(x_index, y_index, wet.T.astype(float), levels=[0.5], colors="#555555", linewidths=0.8)
    ax.set_title("Initial lake-bed depth")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    plt.colorbar(im, ax=ax, shrink=0.82, label="depth (m)")
    fig.tight_layout()
    fig.savefig(output_dir / "initial_bathymetry.png")
    plt.close(fig)


def plot_mean_std_maps(result: SimulationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    u, v = velocity_components(result)
    fields = {
        "zeta": result.zeta,
        "u": u,
        "v": v,
    }
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), dpi=160)
    for row, (name, field) in enumerate(fields.items()):
        mean_field, std_field = masked_stats(field, result.wet_mask)
        _imshow(axes[row, 0], mean_field, result, f"{name} mean")
        _imshow(axes[row, 1], std_field, result, f"{name} standard deviation", cmap="magma")
    fig.suptitle(f"{result.name}: mean and standard deviation maps")
    fig.tight_layout()
    fig.savefig(output_dir / f"{result.name}_mean_std_maps.png")
    plt.close(fig)


def plot_hovmoller(results: list[SimulationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_index = results[0].config.transect_x_index
    # Hovmoller view fixes x and shows zeta evolving over y and time.
    transects = [np.where(result.wet_mask[x_index, :], result.zeta[:, x_index, :], np.nan) for result in results]
    zmax = max(float(np.nanmax(np.abs(transect))) for transect in transects)
    if zmax == 0.0:
        zmax = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180)
    axes = axes.ravel()
    for ax, result, transect in zip(axes, results, transects):
        im = ax.imshow(
            transect,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(-0.5, result.depth.shape[1] - 0.5, result.steps[0], result.steps[-1]),
            cmap="coolwarm",
            vmin=-zmax,
            vmax=zmax,
        )
        ax.set_title(result.name)
        ax.set_xlabel("y index")
        ax.set_ylabel("model step index")
        ax.set_xticks(np.arange(0, result.depth.shape[1], 2))
        ax.set_yticks(np.linspace(result.steps[0], result.steps[-1], 6, dtype=int))
        plt.colorbar(im, ax=ax, shrink=0.82, label="zeta (m)")
    fig.suptitle(f"Hovmoller plots for zeta along x index = {x_index}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_flow_snapshots(result: SimulationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use representative saved frames rather than every model step for a compact summary figure.
    frame_indices = sorted({0, len(result.times) // 3, 2 * len(result.times) // 3, len(result.times) - 1})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), dpi=160)
    axes = axes.ravel()
    for ax, idx in zip(axes, frame_indices):
        title = f"step={result.steps[idx]}, t={result.times[idx] / 3600:.2f} h\n{_wind_label(result, idx)}"
        _plot_stream_snapshot(ax, result, idx, title)
    scenario_title = textwrap.fill(result.description, width=92)
    fig.suptitle(f"{result.name}: flow snapshots\n{scenario_title}", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / f"{result.name}_flow_snapshots.png")
    plt.close(fig)

def plot_presentation_style(result: SimulationResult, output_dir: Path) -> None:
    """Create a cleaner slide-style figure inspired by the lecture screenshot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    early = max(1, len(result.times) // 4)
    late = len(result.times) - 1
    transport = net_transport_through_channel(result)

    fig = plt.figure(figsize=(16, 8.5), dpi=160)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.12, 0.88],
        width_ratios=[1.0, 1.0, 1.45],
        left=0.045,
        right=0.98,
        top=0.94,
        bottom=0.08,
        wspace=0.26,
    )

    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.55,
        "The Semester Project: Model of a Lake",
        fontsize=24,
        fontweight="bold",
        va="center",
        ha="left",
    )
    title_ax.text(
        1.0,
        0.58,
        "TUM",
        fontsize=20,
        fontweight="bold",
        color="#0065bd",
        va="center",
        ha="right",
    )

    ax1 = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[1, 1])
    ax3 = fig.add_subplot(grid[1, 2])

    _plot_stream_snapshot(
        ax1,
        result,
        early,
        f"{result.name}: step={result.steps[early]}",
    )
    _plot_stream_snapshot(
        ax2,
        result,
        late,
        f"{result.name}: step={result.steps[late]}",
    )

    ax3.plot(result.steps, transport, linewidth=2.0)
    ax3.set_title("Net Transport through Channel", fontsize=17)
    ax3.set_xlabel("model step")
    ax3.set_ylabel("transport")
    ax3.grid(True, alpha=0.35)

    fig.text(0.30, 0.035, "Velocity snapshots!", fontsize=17, ha="center")
    fig.text(
        0.045,
        0.025,
        "Deutsches Geodätisches Forschungsinstitut (DGFI-TUM) | Technische Universität München",
        fontsize=11,
        color="#0065bd",
        ha="left",
    )

    fig.savefig(output_dir / f"{result.name}_presentation_style.png")
    plt.close(fig)


def animate_lake(result: SimulationResult, output_dir: Path, max_frames: int = 90) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = np.linspace(
        0,
        len(result.times) - 1,
        min(max_frames, len(result.times)),
        dtype=int,
    )

    x_index, y_index = _lake_axes(result)
    u, v = velocity_components(result)
    zeta = np.where(result.wet_mask, result.zeta, np.nan)

    zmax = float(np.nanmax(np.abs(zeta)))
    if zmax == 0.0:
        zmax = 1.0

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=120)

    im = ax.imshow(
        zeta[0].T,
        origin="lower",
        extent=_extent(result),
        aspect="equal",
        cmap="coolwarm",
        vmin=-zmax,
        vmax=zmax,
        interpolation="bilinear",
    )

    ax.contour(
        x_index,
        y_index,
        result.wet_mask.T.astype(float),
        levels=[0.5],
        colors="#333333",
        linewidths=0.8,
    )

    stride = 2
    xx, yy = np.meshgrid(x_index, y_index)

    quiver = ax.quiver(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        u[0].T[::stride, ::stride],
        v[0].T[::stride, ::stride],
        color="black",
        scale=0.04,
        width=0.003,
    )

    title = ax.set_title("")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    plt.colorbar(im, ax=ax, shrink=0.78, label="zeta (m)")

    def update(frame_id: int):
        im.set_data(zeta[frame_id].T)
        quiver.set_UVC(
            u[frame_id].T[::stride, ::stride],
            v[frame_id].T[::stride, ::stride],
        )
        wx, wy = result.winds[frame_id]
        title.set_text(
            f"{result.name} | step={result.steps[frame_id]} | wind=({wx:.0f}, {wy:.0f}) m/s"
        )
        return im, quiver, title

    animation = matplotlib.animation.FuncAnimation(
        fig,
        update,
        frames=frame_ids,
        interval=90,
        blit=False,
    )

    animation.save(
        output_dir / f"{result.name}_lake_animation.gif",
        writer=PillowWriter(fps=12),
    )

    plt.close(fig)


def animate_scenario_comparison(
    results: list[SimulationResult],
    output_dir: Path,
    max_frames: int = 90,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = min(max_frames, *(len(result.times) for result in results))
    frame_ids = np.linspace(
        0,
        min(len(result.times) for result in results) - 1,
        frame_count,
        dtype=int,
    )

    zmax = max(
        float(np.nanmax(np.abs(np.where(result.wet_mask, result.zeta, np.nan))))
        for result in results
    )
    if zmax == 0.0:
        zmax = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), dpi=120)
    axes = axes.ravel()

    images = []
    for ax, result in zip(axes, results):
        x_index, y_index = _lake_axes(result)

        im = ax.imshow(
            np.where(result.wet_mask, result.zeta[0], np.nan).T,
            origin="lower",
            extent=_extent(result),
            aspect="equal",
            cmap="coolwarm",
            vmin=-zmax,
            vmax=zmax,
            interpolation="bilinear",
        )

        ax.contour(
            x_index,
            y_index,
            result.wet_mask.T.astype(float),
            levels=[0.5],
            colors="#333333",
            linewidths=0.7,
        )

        ax.set_title(result.name)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        images.append(im)

    fig.colorbar(images[0], ax=axes.tolist(), shrink=0.72, label="zeta (m)")
    fig.suptitle("Lake sea-level response through time")

    def update(frame_id: int):
        for im, result in zip(images, results):
            im.set_data(np.where(result.wet_mask, result.zeta[frame_id], np.nan).T)

        fig.suptitle(
            f"Lake sea-level response through time | step={results[0].steps[frame_id]}"
        )
        return images

    animation = matplotlib.animation.FuncAnimation(
        fig,
        update,
        frames=frame_ids,
        interval=90,
        blit=False,
    )

    animation.save(
        output_dir / "all_scenarios_lake_animation.gif",
        writer=PillowWriter(fps=12),
    )

    plt.close(fig)


def animate_4d_surface(result: SimulationResult, output_dir: Path, max_frames: int = 90) -> None:
    """
    4D-style lake visualization:
    x/y position = lake grid
    z axis = sea surface elevation zeta
    color = velocity magnitude
    animation = time
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = np.linspace(
        0,
        len(result.times) - 1,
        min(max_frames, len(result.times)),
        dtype=int,
    )

    x_index, y_index = _lake_axes(result)
    X, Y = np.meshgrid(x_index, y_index)

    u, v = velocity_components(result)
    speed = np.sqrt(u**2 + v**2)

    zeta = np.where(result.wet_mask, result.zeta, np.nan)
    speed = np.where(result.wet_mask, speed, np.nan)

    zmax = float(np.nanmax(np.abs(zeta)))
    if zmax == 0.0:
        zmax = 1.0

    speed_max = float(np.nanmax(speed))
    if speed_max == 0.0:
        speed_max = 1.0

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0.0, vmax=speed_max)

    fig = plt.figure(figsize=(8, 7), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    def draw_frame(frame_id: int):
        ax.clear()

        Z = zeta[frame_id].T
        C = cmap(norm(speed[frame_id].T))

        ax.plot_surface(
            X,
            Y,
            Z,
            facecolors=C,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=False,
        )

        wx, wy = result.winds[frame_id]

        ax.set_title(
            f"{result.name} | 4D lake surface | step={result.steps[frame_id]}\n"
            f"wind=({wx:.0f}, {wy:.0f}) m/s"
        )
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        ax.set_zlabel("zeta (m)")
        ax.set_zlim(-zmax, zmax)
        ax.view_init(elev=35, azim=-55)

        return []

    draw_frame(frame_ids[0])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.65, label="velocity magnitude (m/s)")

    animation = matplotlib.animation.FuncAnimation(
        fig,
        draw_frame,
        frames=frame_ids,
        interval=90,
        blit=False,
    )

    animation.save(
        output_dir / f"{result.name}_4d_surface_animation.gif",
        writer=PillowWriter(fps=12),
    )

    plt.close(fig)

def plot_vorticity_eke(results: list[SimulationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [result.name for result in results]
    mean_vort = [float(np.nanmean(vorticity(result))) for result in results]
    mean_eke = [float(np.nanmean(eddy_kinetic_energy(result))) for result in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=180)
    axes[0].bar(names, mean_vort, color="#3B7EA1")
    axes[0].set_title("Mean vorticity")
    axes[0].set_ylabel("s^-1")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(names, mean_eke, color="#B85C38")
    axes[1].set_title("Mean eddy kinetic energy")
    axes[1].set_ylabel("m^2 s^-2")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_all(results: list[SimulationResult], output_dir: Path, make_animations: bool = False) -> None:
    figures_dir = output_dir / "figures"
    animations_dir = output_dir / "animations"

    plot_input_maps(results[0].depth, figures_dir)
    plot_point_timeseries(results, figures_dir / "question_a_point_timeseries.png")
    plot_hovmoller(results, figures_dir / "question_c_hovmoller_transect_25.png")
    plot_vorticity_eke(results, figures_dir / "question_e_vorticity_eke.png")

    for result in results:
        plot_mean_std_maps(result, figures_dir)
        plot_flow_snapshots(result, figures_dir)
        plot_presentation_style(result, figures_dir)

        if make_animations:
            animate_lake(result, animations_dir)
            animate_4d_surface(result, animations_dir)

    if make_animations:
        animate_scenario_comparison(results, animations_dir)


