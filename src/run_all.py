from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import ModelConfig, load_bathymetry
from .diagnostics import save_npz, save_summary_csv
from .model import run_simulation
from .plots import plot_all
from .scenarios import SCENARIOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SWE lake model scenarios.")
    parser.add_argument("--steps", type=int, default=None, help="Number of model time steps.")
    parser.add_argument("--output-every", type=int, default=None, help="Save every Nth time step.")
    parser.add_argument("--dx", type=float, default=None, help="Grid spacing in x direction, meters.")
    parser.add_argument("--dy", type=float, default=None, help="Grid spacing in y direction, meters.")
    parser.add_argument("--dt", type=float, default=None, help="Time step in seconds.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for figures and data.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ModelConfig:
    config = ModelConfig()
    # Keep dataclass defaults unless the CLI explicitly overrides them.
    replacements = {}
    if args.steps is not None:
        replacements["steps"] = args.steps
    if args.output_every is not None:
        replacements["output_every"] = args.output_every
    if args.dx is not None:
        replacements["dx"] = args.dx
    if args.dy is not None:
        replacements["dy"] = args.dy
    if args.dt is not None:
        replacements["dt"] = args.dt
    if args.output_dir is not None:
        replacements["output_dir"] = args.output_dir
    return replace(config, **replacements)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    depth = load_bathymetry(config.bathymetry_path)

    print("SWE lake model")
    print(f"bathymetry: {config.bathymetry_path}")
    print(f"grid shape: {depth.shape}, max depth: {depth.max():.1f} m")
    print(f"dx={config.dx:.1f} m, dy={config.dy:.1f} m, dt={config.dt:.3f} s")
    print(f"steps={config.steps}, output_every={config.output_every}")

    results = []
    data_dir = config.output_dir / "data"
    for scenario in SCENARIOS:
        print(f"running {scenario.name}: {scenario.description}")
        result = run_simulation(scenario, depth, config)
        # Save raw arrays before plotting so figures can be regenerated or replayed later.
        save_npz(result, data_dir / f"{scenario.name}.npz")
        results.append(result)

    save_summary_csv(results, config.output_dir / "summary.csv")
    plot_all(results, config.output_dir, make_animations=True)
    print(f"done. outputs written to {config.output_dir}")


if __name__ == "__main__":
    main()
