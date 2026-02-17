"""Tests for lateral envelope convergence behavior."""

from __future__ import annotations

import unittest
from pathlib import Path

from lap_time_sim.simulation.config import NumericsConfig, RuntimeConfig, SimulationConfig
from lap_time_sim.simulation.profile import solve_speed_profile
from lap_time_sim.tire.models import default_axle_tire_parameters
from lap_time_sim.track.io import load_track_csv
from lap_time_sim.vehicle import BicycleModel, BicycleNumerics, BicyclePhysics
from tests.helpers import sample_vehicle_parameters


class SpeedProfileConvergenceTests(unittest.TestCase):
    """Validate configurable early stopping of the lateral envelope loop."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load shared track/model fixtures for convergence tests."""
        root = Path(__file__).resolve().parents[2]
        cls.track = load_track_csv(root / "data" / "spa_francorchamps.csv")
        cls.model = BicycleModel(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=BicyclePhysics(
                max_drive_accel=8.0,
                max_brake_accel=16.0,
                peak_slip_angle=0.12,
            ),
            numerics=BicycleNumerics(
                min_lateral_accel_limit=0.5,
                lateral_limit_max_iterations=12,
                lateral_limit_convergence_tolerance=0.05,
            ),
        )

    def test_large_tolerance_converges_in_single_iteration(self) -> None:
        """Converge after one update when tolerance is intentionally loose."""
        config = SimulationConfig(
            runtime=RuntimeConfig(
                max_speed=115.0,
                enable_transient_refinement=False,
            ),
            numerics=NumericsConfig(
                lateral_envelope_max_iterations=20,
                lateral_envelope_convergence_tolerance=1_000.0,
                min_speed=8.0,
                transient_step=0.01,
            )
        )
        result = solve_speed_profile(self.track, self.model, config)
        self.assertEqual(result.lateral_envelope_iterations, 1)

    def test_small_tolerance_requires_multiple_iterations(self) -> None:
        """Require multiple iterations when tolerance is tight."""
        config = SimulationConfig(
            runtime=RuntimeConfig(
                max_speed=115.0,
                enable_transient_refinement=False,
            ),
            numerics=NumericsConfig(
                lateral_envelope_max_iterations=20,
                lateral_envelope_convergence_tolerance=0.01,
                min_speed=8.0,
                transient_step=0.01,
            )
        )
        result = solve_speed_profile(self.track, self.model, config)
        self.assertGreater(result.lateral_envelope_iterations, 1)
        self.assertLessEqual(
            result.lateral_envelope_iterations,
            config.numerics.lateral_envelope_max_iterations,
        )


if __name__ == "__main__":
    unittest.main()
