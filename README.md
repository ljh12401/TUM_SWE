# SWE Lake Model Course Project

This project implements the semester assignment Shallow Water Equation: a four-scenario lake model using the linearized shallow water equations and the supplied bathymetry grid.

## What Is Included

- `src/config.py`: physical constants, grid defaults, and manually selected time step.
- `src/scenarios.py`: the four wind/barrier scenarios.
- `src/model.py`: finite-difference shallow water solver for `zeta`, `U`, and `V`.
- `src/diagnostics.py`: velocity, vorticity, eddy kinetic energy, summary CSV, data export.
- `src/plots.py`: figures for the assignment questions.
- `tests/test_project.py`: data, scenario, stability, and output checks.

## Assumptions

- `bathymetry.txt` has shape `(40, 20)`.
- Depth `H=0` means land or closed boundary.
- Grid indices are Python zero-based indices. The required point `[25, 10]` is read as `zeta[:, 25, 10]`.
- Plan-view figures use the intuitive plotting convention: `x index` is horizontal and `y index` is vertical. Arrays are stored internally as `(x, y)` and transposed only for display.
- Wind components use positive `WX` eastward and positive `WY` northward. Meteorological easterly wind is represented as `WX=-10`; northward wind is represented as `WY=10`; meteorological northerly wind is represented as `WY=-5`.
- Because no `dx`, `dy`, or `dt` is provided in the lecture PDF, defaults are `dx=dy=1000 m` and `dt=5 s`.
- Saved `.npz` result files include one frame per model step by default, plus the corresponding `steps` array.

## Run Everything

```bash
python3 -m src.run_all
```

Optional overrides:

```bash
python3 -m src.run_all --steps 1000 --output-every 5 --dx 1000 --dy 1000
python3 -m src.run_all --dt 10
```

Replay saved sea-level and streamline flow fields directly from disk:

```bash
python3 -m src.replay_zeta outputs/data/scenario_1.npz
```

Render a GIF replay instead of live playback:

```bash
python3 -m src.replay_zeta outputs/data/scenario_1.npz --save-gif outputs/animations/scenario_1_replay.gif
```

The run writes:

- `outputs/summary.csv`
- `outputs/data/scenario_*.npz`
- `outputs/figures/question_a_point_timeseries.png`
- `outputs/figures/question_c_hovmoller_transect_25.png`
- `outputs/figures/question_e_vorticity_eke.png`
- `outputs/figures/scenario_*_mean_std_maps.png`
- `outputs/figures/scenario_*_flow_snapshots.png`

## Assignment Question Mapping

- a) `question_a_point_timeseries.png` shows sea level `zeta` at `[25, 10]`.
- b) `scenario_*_mean_std_maps.png` shows mean and standard deviation of `zeta`, `u`, and `v`.
- c) `question_c_hovmoller_transect_25.png` compares `zeta` along transect `x=25`.
- d) `scenario_*_flow_snapshots.png` shows flow snapshots over time.
- e) `question_e_vorticity_eke.png` and `summary.csv` compare mean vorticity and eddy kinetic energy.

## Run Tests

```bash
python3 -m unittest discover -s tests
```
