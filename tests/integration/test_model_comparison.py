"""Integration tests for bicycle versus point-mass model comparison."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from lap_time_sim.simulation import build_simulation_config, simulate_lap
from lap_time_sim.tire import default_axle_tire_parameters
from lap_time_sim.track import load_track_csv
from lap_time_sim.vehicle import (
    BicyclePhysics,
    PointMassPhysics,
    build_bicycle_model,
    build_point_mass_model,
)
from tests.helpers import sample_vehicle_parameters


class ModelComparisonTests(unittest.TestCase):
    """Validate plug-and-play behavior across multiple vehicle-model backends."""

    def test_bicycle_and_point_mass_models_both_run_in_same_solver(self) -> None:
        """Run both backends on the same track and compare key output invariants."""
        root = Path(__file__).resolve().parents[2]
        track = load_track_csv(root / "data" / "spa_francorchamps.csv")
        vehicle = sample_vehicle_parameters()
        config = build_simulation_config()

        bicycle_model = build_bicycle_model(
            vehicle=vehicle,
            tires=default_axle_tire_parameters(),
            physics=BicyclePhysics(),
        )
        point_mass_model = build_point_mass_model(
            vehicle=vehicle,
            physics=PointMassPhysics(),
        )

        bicycle_result = simulate_lap(track=track, model=bicycle_model, config=config)
        point_mass_result = simulate_lap(track=track, model=point_mass_model, config=config)

        self.assertTrue(np.isfinite(bicycle_result.lap_time_s))
        self.assertTrue(np.isfinite(point_mass_result.lap_time_s))
        self.assertGreater(bicycle_result.lap_time_s, 60.0)
        self.assertGreater(point_mass_result.lap_time_s, 60.0)
        self.assertEqual(len(bicycle_result.speed_mps), len(point_mass_result.speed_mps))
        self.assertTrue(np.max(np.abs(point_mass_result.yaw_moment_nm)) == 0.0)
        self.assertGreater(np.max(np.abs(bicycle_result.yaw_moment_nm)), 1e-9)


if __name__ == "__main__":
    unittest.main()
