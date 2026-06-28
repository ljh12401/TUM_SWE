from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.animation import PillowWriter
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved lake-model zeta and flow fields from a .npz or .npy file.")
    parser.add_argument("input", type=Path, help="Saved result file, e.g. outputs/data/scenario_1.npz.")
    parser.add_argument("--depth", type=Path, default=None, help="Bathymetry file for masking land when input is .npy.")
    parser.add_argument("--interval", type=float, default=0.03, help="Pause between frames in seconds.")
    parser.add_argument("--start-step", type=int, default=1, help="First model step to display.")
    parser.add_argument("--end-step", type=int, default=None, help="Last model step to display.")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth saved frame in the selected step range.")
    parser.add_argument("--stream-density", type=float, default=0.95, help="Density of flow streamlines.")
    parser.add_argument("--stream-linewidth", type=float, default=0.85, help="Width of flow streamlines.")
    parser.add_argument("--stream-arrowsize", type=float, default=0.8, help="Arrow size for flow streamlines.")
    parser.add_argument("--save-gif", type=Path, default=None, help="Render the replay to a GIF instead of live playback.")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second when --save-gif is used.")
    parser.add_argument("--no-block", action="store_true", help="Exit immediately after the replay finishes.")
    return parser.parse_args()


def _load_replay_data(
    path: Path,
    depth_path: Path | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray | None, float | None]:
    loaded = np.load(path)

    if isinstance(loaded, np.lib.npyio.NpzFile):
        # Preferred replay format: includes zeta, transports, depth, and original model steps.
        zeta = loaded["zeta"]
        U = loaded["U"] if "U" in loaded.files else None
        V = loaded["V"] if "V" in loaded.files else None
        steps = loaded["steps"] if "steps" in loaded.files else np.arange(zeta.shape[0])
        depth = loaded["depth"] if "depth" in loaded.files else None
        winds = loaded["winds"] if "winds" in loaded.files else None
        dt = float(loaded["dt"]) if "dt" in loaded.files else None
        return zeta, U, V, steps, depth, winds, dt

    zeta = np.asarray(loaded)
    if zeta.ndim != 3:
        raise ValueError(".npy input must have shape (frames, x, y).")

    # A bare .npy file can replay zeta only; depth is optional and used only for masking land.
    depth = np.loadtxt(depth_path, dtype=float) if depth_path is not None else None
    steps = np.arange(zeta.shape[0])
    return zeta, None, None, steps, depth, None, None


def _frame_range(steps: np.ndarray, start_step: int, end_step: int | None) -> np.ndarray:
    upper = int(steps[-1]) if end_step is None else end_step
    frame_ids = np.flatnonzero((steps >= start_step) & (steps <= upper))
    if frame_ids.size == 0:
        raise ValueError(f"No saved frames found between step {start_step} and {upper}.")
    return frame_ids


def main() -> None:
    args = parse_args()
    zeta, U, V, steps, depth, winds, dt = _load_replay_data(args.input, args.depth)

    wet_mask = depth > 0.0 if depth is not None else np.isfinite(zeta[0])
    zeta = np.where(wet_mask, zeta, np.nan)
    if U is not None and V is not None and depth is not None:
        safe_depth = np.where(wet_mask, depth, np.nan)
        u = np.where(wet_mask, U / safe_depth, np.nan)
        v = np.where(wet_mask, V / safe_depth, np.nan)
    else:
        u = None
        v = None

    zmax = float(np.nanmax(np.abs(zeta)))
    if zmax == 0.0:
        zmax = 1.0

    frame_ids = _frame_range(steps, args.start_step, args.end_step)
    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1.")
    # Keep full-frame GIFs possible while allowing faster sampled previews.
    frame_ids = frame_ids[:: args.frame_step]
    nx, ny = zeta.shape[1:]
    x_index = np.arange(nx)
    y_index = np.arange(ny)

    plt.ion()
    fig, (zeta_ax, flow_ax) = plt.subplots(1, 2, figsize=(12.0, 5.4), dpi=120)
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.09, top=0.9, wspace=0.22)

    image = zeta_ax.imshow(
        zeta[frame_ids[0]].T,
        origin="lower",
        extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
        aspect="equal",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-zmax,
        vmax=zmax,
    )
    if depth is not None:
        zeta_ax.contour(x_index, y_index, wet_mask.T.astype(float), levels=[0.5], colors="#555555", linewidths=0.8)
    zeta_ax.set_title("zeta")
    zeta_ax.set_xlabel("x index")
    zeta_ax.set_ylabel("y index")
    colorbar = fig.colorbar(image, ax=zeta_ax, shrink=0.82)
    colorbar.set_label("zeta (m)")

    def draw_flow_base() -> None:
        # streamplot creates new artists every frame, so redraw the flow axis from a clean base.
        flow_ax.clear()
        flow_ax.contourf(x_index, y_index, wet_mask.T.astype(float), levels=[0.5, 1.5], colors=["#f7fbff"], alpha=1.0)
        if depth is not None:
            flow_ax.contour(x_index, y_index, wet_mask.T.astype(float), levels=[0.5], colors="#555555", linewidths=0.8)
        flow_ax.set_title("flow")
        flow_ax.set_xlabel("x index")
        flow_ax.set_ylabel("y index")
        flow_ax.set_xlim(x_index.min(), x_index.max())
        flow_ax.set_ylim(y_index.min(), y_index.max())
        flow_ax.set_aspect("equal", adjustable="box")

    draw_flow_base()

    if u is not None and v is not None:
        speed = np.sqrt(u * u + v * v)
        speed_max = float(np.nanmax(speed))
        if speed_max == 0.0:
            speed_max = 1.0
        # Fixed normalization prevents the velocity colorbar from changing during animation.
        speed_norm = Normalize(vmin=0.0, vmax=speed_max)
        flow_colorbar = fig.colorbar(
            ScalarMappable(norm=speed_norm, cmap="Spectral_r"),
            ax=flow_ax,
            shrink=0.82,
        )
        flow_colorbar.set_label("velocity (m/s)")
    else:
        speed = None
        speed_norm = None
        flow_ax.text(
            0.5,
            0.5,
            "flow unavailable\nuse .npz with U, V, and depth",
            transform=flow_ax.transAxes,
            ha="center",
            va="center",
        )

    title = fig.suptitle("")

    def frame_title(frame_id: int) -> str:
        step = int(steps[frame_id])
        time_label = f", t={step * dt / 3600:.2f} h" if dt is not None else ""
        if winds is not None:
            wx, wy = winds[frame_id]
            wind_label = f"wind=({wx:.0f}, {wy:.0f}) m/s"
        else:
            wind_label = "wind unavailable"
        return f"{args.input.stem} | step={step}{time_label} | {wind_label}"

    def update(frame_id: int):
        image.set_data(zeta[frame_id].T)
        if u is not None and v is not None and speed is not None and speed_norm is not None:
            draw_flow_base()
            flow_ax.streamplot(
                x_index,
                y_index,
                np.ma.array(u[frame_id].T, mask=~wet_mask.T),
                np.ma.array(v[frame_id].T, mask=~wet_mask.T),
                color=np.ma.array(speed[frame_id].T, mask=~wet_mask.T),
                cmap="Spectral_r",
                norm=speed_norm,
                density=args.stream_density,
                linewidth=args.stream_linewidth,
                arrowsize=args.stream_arrowsize,
            )
        title.set_text(frame_title(frame_id))
        return (image,)

    if args.save_gif is not None:
        args.save_gif.parent.mkdir(parents=True, exist_ok=True)
        replay = animation.FuncAnimation(
            fig,
            update,
            frames=frame_ids,
            interval=1000 / max(1, args.fps),
            blit=False,
        )
        replay.save(args.save_gif, writer=PillowWriter(fps=args.fps))
        print(f"Saved {frame_ids.size} replay frames to {args.save_gif}.")
        plt.close(fig)
        return

    for frame_id in frame_ids:
        update(frame_id)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        if args.interval > 0.0:
            plt.pause(args.interval)
        else:
            fig.canvas.draw()

    print(f"Replayed {frame_ids.size} saved frames from {args.input}.")
    if not args.no_block:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
