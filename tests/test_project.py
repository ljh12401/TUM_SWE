from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.config import ModelConfig, load_bathymetry
from src.model import run_simulation
from src.plots import plot_all
from src.scenarios import SCENARIOS, apply_artificial_barrier


class SweProjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig(steps=10, output_every=1)
        self.depth = load_bathymetry(self.config.bathymetry_path)

    def test_bathymetry_shape_and_values(self) -> None:
        self.assertEqual(self.depth.shape, (40, 20))
        self.assertEqual(float(self.depth.max()), 40.0)
        self.assertEqual(set(np.unique(self.depth)), {0.0, 10.0, 20.0, 30.0, 40.0})

    def test_wind_scenarios(self) -> None:
        self.assertEqual(SCENARIOS[0].wind(0), (-10.0, 0.0))
        self.assertEqual(SCENARIOS[1].wind(0), (0.0, 10.0))
        self.assertEqual(SCENARIOS[1].wind(50), (-10.0, 0.0))
        self.assertEqual(SCENARIOS[1].wind(250), (0.0, 0.0))
        self.assertEqual(SCENARIOS[2].wind(49), (-10.0, -5.0))
        self.assertEqual(SCENARIOS[2].wind(50), (-10.0, 0.0))
        self.assertEqual(SCENARIOS[2].wind(249), (-10.0, 0.0))
        self.assertEqual(SCENARIOS[2].wind(250), (0.0, 0.0))

    def test_barrier_sets_middle_x_to_land(self) -> None:
        barrier_depth = apply_artificial_barrier(self.depth)
        middle_x = self.depth.shape[0] // 2
        self.assertTrue(np.all(barrier_depth[middle_x, :] == 0.0))
        self.assertEqual(barrier_depth.shape, self.depth.shape)

    def test_short_run_is_finite_and_keeps_land_masked(self) -> None:
        result = run_simulation(SCENARIOS[0], self.depth, self.config)
        self.assertTrue(np.all(np.isfinite(result.zeta)))
        self.assertTrue(np.all(np.isfinite(result.U)))
        self.assertTrue(np.all(np.isfinite(result.V)))
        self.assertTrue(np.all(result.zeta[:, self.depth == 0.0] == 0.0))

    def test_output_figures_are_created(self) -> None:
        config = replace(self.config, steps=5, output_every=1)
        results = [run_simulation(scenario, self.depth, config) for scenario in SCENARIOS]
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            plot_all(results, out)
            expected = [
                out / "figures" / "question_a_point_timeseries.png",
                out / "figures" / "question_c_hovmoller_transect_25.png",
                out / "figures" / "question_e_vorticity_eke.png",
            ]
            for path in expected:
                self.assertTrue(path.exists(), f"Missing output figure {path}")


if __name__ == "__main__":
    unittest.main()
