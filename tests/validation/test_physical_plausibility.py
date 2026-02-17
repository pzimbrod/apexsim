"""Validation tests for physical plausibility boundaries."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from lap_time_sim.analysis.kpi import compute_kpis
from lap_time_sim.simulation.config import NumericsConfig, RuntimeConfig, SimulationConfig
from lap_time_sim.simulation.runner import simulate_lap
from lap_time_sim.tire.models import default_axle_tire_parameters
from lap_time_sim.track.io import load_track_csv
from lap_time_sim.vehicle import BicycleModel, BicycleNumerics, BicyclePhysics
from tests.helpers import sample_vehicle_parameters


class PhysicalValidationTests(unittest.TestCase):
    """Plausibility checks against realistic racing ranges."""

    def test_acceleration_ranges_stay_realistic(self) -> None:
        """Keep simulated acceleration metrics within plausible racing bounds."""
        root = Path(__file__).resolve().parents[2]
        track = load_track_csv(root / "data" / "spa_francorchamps.csv")
        model = BicycleModel(
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

        result = simulate_lap(
            track=track,
            model=model,
            config=SimulationConfig(
                runtime=RuntimeConfig(
                    max_speed=115.0,
                    enable_transient_refinement=False,
                ),
                numerics=NumericsConfig(
                    min_speed=8.0,
                    lateral_envelope_max_iterations=20,
                    lateral_envelope_convergence_tolerance=0.1,
                    transient_step=0.01,
                ),
            ),
        )
        kpis = compute_kpis(result)

        self.assertLess(kpis.max_lateral_accel_g, 8.5)
        self.assertLess(kpis.max_longitudinal_accel_g, 5.0)
        self.assertGreater(kpis.avg_lateral_accel_g, 0.6)
        self.assertTrue(np.isfinite(result.lap_time))


if __name__ == "__main__":
    unittest.main()
