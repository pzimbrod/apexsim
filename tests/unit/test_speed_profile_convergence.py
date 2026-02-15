"""Tests for lateral envelope convergence behavior."""

from __future__ import annotations

import unittest
from pathlib import Path

from lap_time_sim.simulation.config import SimulationConfig
from lap_time_sim.simulation.profile import solve_speed_profile
from lap_time_sim.tire.models import default_axle_tire_parameters
from lap_time_sim.track.io import load_track_csv
from lap_time_sim.vehicle import BicycleLapTimeModel
from lap_time_sim.vehicle.params import default_vehicle_parameters


class SpeedProfileConvergenceTests(unittest.TestCase):
    """Validate configurable early stopping of the lateral envelope loop."""

    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[2]
        cls.track = load_track_csv(root / "data" / "spa_francorchamps.csv")
        cls.model = BicycleLapTimeModel(
            vehicle=default_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
        )

    def test_large_tolerance_converges_in_single_iteration(self) -> None:
        config = SimulationConfig(
            lateral_envelope_max_iterations=20,
            lateral_envelope_convergence_tol_mps=1_000.0,
        )
        result = solve_speed_profile(self.track, self.model, config)
        self.assertEqual(result.lateral_envelope_iterations, 1)

    def test_small_tolerance_requires_multiple_iterations(self) -> None:
        config = SimulationConfig(
            lateral_envelope_max_iterations=20,
            lateral_envelope_convergence_tol_mps=0.01,
        )
        result = solve_speed_profile(self.track, self.model, config)
        self.assertGreater(result.lateral_envelope_iterations, 1)
        self.assertLessEqual(
            result.lateral_envelope_iterations,
            config.lateral_envelope_max_iterations,
        )


if __name__ == "__main__":
    unittest.main()
