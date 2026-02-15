"""Validation tests for physical plausibility boundaries."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from lap_time_sim.analysis.kpi import compute_kpis
from lap_time_sim.simulation.runner import simulate_lap
from lap_time_sim.tire.models import default_axle_tire_parameters
from lap_time_sim.track.io import load_track_csv
from lap_time_sim.vehicle import BicycleLapTimeModel
from lap_time_sim.vehicle.params import default_vehicle_parameters


class PhysicalValidationTests(unittest.TestCase):
    """Plausibility checks against realistic racing ranges."""

    def test_acceleration_ranges_stay_realistic(self) -> None:
        """Keep simulated acceleration metrics within plausible racing bounds."""
        root = Path(__file__).resolve().parents[2]
        track = load_track_csv(root / "data" / "spa_francorchamps.csv")
        model = BicycleLapTimeModel(
            vehicle=default_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
        )

        result = simulate_lap(track=track, model=model)
        kpis = compute_kpis(result)

        self.assertLess(kpis.max_lateral_accel_g, 8.5)
        self.assertLess(kpis.max_longitudinal_accel_g, 5.0)
        self.assertGreater(kpis.avg_lateral_accel_g, 0.6)
        self.assertTrue(np.isfinite(result.lap_time_s))


if __name__ == "__main__":
    unittest.main()
