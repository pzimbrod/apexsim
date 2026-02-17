"""Integration tests for bicycle versus point-mass model comparison."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from pylapsim.simulation import build_simulation_config, simulate_lap
from pylapsim.tire import default_axle_tire_parameters
from pylapsim.track import load_track_csv
from pylapsim.vehicle import (
    BicyclePhysics,
    PointMassPhysics,
    build_bicycle_model,
    build_point_mass_model,
    calibrate_point_mass_friction_to_bicycle,
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

        self.assertTrue(np.isfinite(bicycle_result.lap_time))
        self.assertTrue(np.isfinite(point_mass_result.lap_time))
        self.assertGreater(bicycle_result.lap_time, 60.0)
        self.assertGreater(point_mass_result.lap_time, 60.0)
        self.assertEqual(len(bicycle_result.speed), len(point_mass_result.speed))
        self.assertTrue(np.max(np.abs(point_mass_result.yaw_moment)) == 0.0)
        self.assertGreater(np.max(np.abs(bicycle_result.yaw_moment)), 1e-9)

    def test_calibrated_point_mass_matches_bicycle_more_closely(self) -> None:
        """Reduce bicycle-vs-point-mass speed and lap-time deltas via calibration."""
        root = Path(__file__).resolve().parents[2]
        track = load_track_csv(root / "data" / "spa_francorchamps.csv")
        vehicle = sample_vehicle_parameters()
        config = build_simulation_config()
        tires = default_axle_tire_parameters()
        bicycle_physics = BicyclePhysics()

        bicycle_model = build_bicycle_model(vehicle=vehicle, tires=tires, physics=bicycle_physics)
        bicycle_result = simulate_lap(track=track, model=bicycle_model, config=config)

        point_mass_default = build_point_mass_model(
            vehicle=vehicle,
            physics=PointMassPhysics(
                max_drive_accel=bicycle_physics.max_drive_accel,
                max_brake_accel=bicycle_physics.max_brake_accel,
                friction_coefficient=1.70,
            ),
        )
        default_result = simulate_lap(track=track, model=point_mass_default, config=config)

        calibration = calibrate_point_mass_friction_to_bicycle(
            vehicle=vehicle,
            tires=tires,
            bicycle_physics=bicycle_physics,
            speed_samples=bicycle_result.speed,
        )
        point_mass_calibrated = build_point_mass_model(
            vehicle=vehicle,
            physics=PointMassPhysics(
                max_drive_accel=bicycle_physics.max_drive_accel,
                max_brake_accel=bicycle_physics.max_brake_accel,
                friction_coefficient=calibration.friction_coefficient,
            ),
        )
        calibrated_result = simulate_lap(track=track, model=point_mass_calibrated, config=config)

        lap_delta_default = abs(bicycle_result.lap_time - default_result.lap_time)
        lap_delta_calibrated = abs(bicycle_result.lap_time - calibrated_result.lap_time)
        self.assertLess(lap_delta_calibrated, lap_delta_default)

        segment_mask = (track.arc_length >= 6075.0) & (track.arc_length <= 6280.0)
        default_speed_delta = np.abs(
            bicycle_result.speed[segment_mask] - default_result.speed[segment_mask]
        )
        calibrated_speed_delta = np.abs(
            bicycle_result.speed[segment_mask] - calibrated_result.speed[segment_mask]
        )
        self.assertLess(float(np.mean(calibrated_speed_delta)), float(np.mean(default_speed_delta)))


if __name__ == "__main__":
    unittest.main()
