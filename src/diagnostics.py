from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .model import SimulationResult, gradient_x, gradient_y


def velocity_components(result: SimulationResult) -> tuple[np.ndarray, np.ndarray]:
    """Convert depth-integrated transports U/V into depth-averaged velocities."""

    safe_depth = np.where(result.depth > 0.0, result.depth, np.nan)
    u = result.U / safe_depth
    v = result.V / safe_depth
    return u, v


def masked_stats(field: np.ndarray, wet_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute time statistics only over permanently wet cells."""

    mean_field = np.full(field.shape[1:], np.nan, dtype=float)
    std_field = np.full(field.shape[1:], np.nan, dtype=float)
    mean_field[wet_mask] = np.mean(field[:, wet_mask], axis=0)
    std_field[wet_mask] = np.std(field[:, wet_mask], axis=0)
    return mean_field, std_field


def vorticity(result: SimulationResult) -> np.ndarray:
    """Compute vertical vorticity from depth-averaged velocity."""

    wet_mask = result.wet_mask
    vort = []
    for U, V in zip(result.U, result.V):
        safe_depth = np.where(wet_mask, result.depth, np.nan)
        u = np.nan_to_num(U / safe_depth)
        v = np.nan_to_num(V / safe_depth)
        dvdx = gradient_x(v, result.config.dx, wet_mask)
        dudy = gradient_y(u, result.config.dy, wet_mask)
        vort.append(np.where(wet_mask, dvdx - dudy, np.nan))
    return np.asarray(vort)


def eddy_kinetic_energy(result: SimulationResult) -> np.ndarray:
    """Compute EKE from velocity anomalies relative to each cell's time mean."""

    u, v = velocity_components(result)
    u_mean = np.full(result.depth.shape, np.nan, dtype=float)
    v_mean = np.full(result.depth.shape, np.nan, dtype=float)
    u_mean[result.wet_mask] = np.mean(u[:, result.wet_mask], axis=0)
    v_mean[result.wet_mask] = np.mean(v[:, result.wet_mask], axis=0)
    eke = 0.5 * ((u - u_mean) ** 2 + (v - v_mean) ** 2)
    return np.where(result.wet_mask, eke, np.nan)


def net_transport_through_channel(result: SimulationResult, y_index: int | None = None) -> np.ndarray:
    """Estimate net north-south transport through a y-index channel."""

    if y_index is None:
        y_index = result.depth.shape[1] // 2
    channel_v = np.where(result.depth[:, y_index] > 0.0, result.V[:, :, y_index], 0.0)
    return np.sum(channel_v, axis=1) * result.config.dx


def scenario_summary(result: SimulationResult) -> dict[str, float | str]:
    """Build scalar metrics for the scenario comparison table."""

    u, v = velocity_components(result)
    vort = vorticity(result)
    eke = eddy_kinetic_energy(result)
    zeta_masked = np.where(result.wet_mask, result.zeta, np.nan)
    return {
        "scenario": result.name,
        "description": result.description,
        "dt_seconds": result.dt,
        "saved_frames": result.zeta.shape[0],
        "max_abs_zeta_m": float(np.nanmax(np.abs(zeta_masked))),
        "mean_zeta_m": float(np.nanmean(zeta_masked)),
        "std_zeta_m": float(np.nanstd(zeta_masked)),
        "mean_u_m_s": float(np.nanmean(u)),
        "mean_v_m_s": float(np.nanmean(v)),
        "mean_vorticity_s-1": float(np.nanmean(vort)),
        "mean_eke_m2_s2": float(np.nanmean(eke)),
    }


def save_summary_csv(results: list[SimulationResult], output_path: Path) -> None:
    rows = [scenario_summary(result) for result in results]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_npz(result: SimulationResult, output_path: Path) -> None:
    """Store enough state to replay zeta and flow without rerunning the model."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        depth=result.depth,
        times=result.times,
        steps=result.steps,
        zeta=result.zeta,
        U=result.U,
        V=result.V,
        winds=result.winds,
        dt=result.dt,
    )
