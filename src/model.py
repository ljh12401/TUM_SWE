from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .scenarios import Scenario, apply_artificial_barrier


@dataclass
class SimulationResult:
    name: str
    description: str
    depth: np.ndarray
    times: np.ndarray
    steps: np.ndarray
    zeta: np.ndarray
    U: np.ndarray
    V: np.ndarray
    winds: np.ndarray
    dt: float
    config: ModelConfig

    @property
    def wet_mask(self) -> np.ndarray:
        return self.depth > 0.0


def gradient_x(field: np.ndarray, dx: float, wet_mask: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(field)
    grad[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * dx)
    grad[0, :] = (field[1, :] - field[0, :]) / dx
    grad[-1, :] = (field[-1, :] - field[-2, :]) / dx
    grad[~wet_mask] = 0.0
    return grad


def gradient_y(field: np.ndarray, dy: float, wet_mask: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(field)
    grad[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dy)
    grad[:, 0] = (field[:, 1] - field[:, 0]) / dy
    grad[:, -1] = (field[:, -1] - field[:, -2]) / dy
    grad[~wet_mask] = 0.0
    return grad


def _apply_closed_boundaries(field: np.ndarray, wet_mask: np.ndarray) -> np.ndarray:
    clean = np.where(wet_mask, field, 0.0)
    clean[0, :] = 0.0
    clean[-1, :] = 0.0
    clean[:, 0] = 0.0
    clean[:, -1] = 0.0
    return clean


def step_forward(
    zeta: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    depth: np.ndarray,
    wind_x: float,
    wind_y: float,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wet_mask = depth > 0.0
    safe_depth = np.where(wet_mask, depth, 1.0)

    zeta_x = gradient_x(zeta, config.dx, wet_mask)
    zeta_y = gradient_y(zeta, config.dy, wet_mask)

    transport_speed = np.sqrt(U * U + V * V)
    quadratic_drag = config.friction * transport_speed / (safe_depth * safe_depth)
    wind_speed = float(np.hypot(wind_x, wind_y))

    dUdt = (
        -config.g * depth * zeta_x
        - quadratic_drag * U
        + config.wind_stress * wind_x * wind_speed
        + config.coriolis_f * V
    )
    dVdt = (
        -config.g * depth * zeta_y
        - quadratic_drag * V
        + config.wind_stress * wind_y * wind_speed
        - config.coriolis_f * U
    )

    next_U = _apply_closed_boundaries(U + config.dt * dUdt, wet_mask)
    next_V = _apply_closed_boundaries(V + config.dt * dVdt, wet_mask)

    div_transport = gradient_x(next_U, config.dx, wet_mask) + gradient_y(next_V, config.dy, wet_mask)
    next_zeta = np.where(wet_mask, zeta - config.dt * div_transport, 0.0)

    if not np.all(np.isfinite(next_zeta)) or not np.all(np.isfinite(next_U)) or not np.all(np.isfinite(next_V)):
        raise FloatingPointError("Simulation produced NaN or infinite values.")

    return next_zeta, next_U, next_V


def run_simulation(
    scenario: Scenario,
    depth: np.ndarray,
    config: ModelConfig,
) -> SimulationResult:
    scenario_depth = apply_artificial_barrier(depth) if scenario.add_barrier else depth.copy()
    wet_mask = scenario_depth > 0.0
    shape = scenario_depth.shape

    zeta = np.zeros(shape, dtype=float)
    U = np.zeros(shape, dtype=float)
    V = np.zeros(shape, dtype=float)

    saved_zeta: list[np.ndarray] = []
    saved_U: list[np.ndarray] = []
    saved_V: list[np.ndarray] = []
    saved_times: list[float] = []
    saved_steps: list[int] = []
    saved_winds: list[tuple[float, float]] = []

    for step in range(config.steps + 1):
        if step % config.output_every == 0 or step == config.steps:
            saved_zeta.append(zeta.copy())
            saved_U.append(U.copy())
            saved_V.append(V.copy())
            saved_times.append(step * config.dt)
            saved_steps.append(step)
            saved_winds.append(scenario.wind(min(step, config.steps - 1)))

        if step == config.steps:
            break

        wind_x, wind_y = scenario.wind(step)
        zeta, U, V = step_forward(zeta, U, V, scenario_depth, wind_x, wind_y, config)
        zeta = np.where(wet_mask, zeta, 0.0)
        U = _apply_closed_boundaries(U, wet_mask)
        V = _apply_closed_boundaries(V, wet_mask)

    return SimulationResult(
        name=scenario.name,
        description=scenario.description,
        depth=scenario_depth,
        times=np.asarray(saved_times),
        steps=np.asarray(saved_steps),
        zeta=np.asarray(saved_zeta),
        U=np.asarray(saved_U),
        V=np.asarray(saved_V),
        winds=np.asarray(saved_winds),
        dt=config.dt,
        config=config,
    )
