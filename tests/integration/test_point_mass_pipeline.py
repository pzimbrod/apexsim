"""Integration tests for end-to-end lap simulation with point-mass model."""

from __future__ import annotations

import unittest
from pathlib import Path

from lap_time_sim.analysis.kpi import compute_kpis
from lap_time_sim.simulation import build_simulation_config, simulate_lap
from lap_time_sim.track.io import load_track_csv
from lap_time_sim.vehicle import PointMassPhysics, build_point_mass_model
from tests.helpers import sample_vehicle_parameters


class PointMassPipelineTests(unittest.TestCase):
    """End-to-end system tests for the point-mass backend."""

    def test_full_spa_simulation_runs_with_point_mass_model(self) -> None:
        """Verify that the solver pipeline runs with a point-mass vehicle model."""
        root = Path(__file__).resolve().parents[2]
        track = load_track_csv(root / "data" / "spa_francorchamps.csv")
        model = build_point_mass_model(
            vehicle=sample_vehicle_parameters(),
            physics=PointMassPhysics(
                max_drive_accel=8.0,
                max_brake_accel=16.0,
                friction_coefficient=1.7,
            ),
        )
        result = simulate_lap(track=track, model=model, config=build_simulation_config())
        kpis = compute_kpis(result)

        self.assertGreater(kpis.lap_time_s, 70.0)
        self.assertLess(kpis.lap_time_s, 260.0)
        self.assertGreater(kpis.max_lateral_accel_g, 1.2)
        self.assertLess(kpis.max_lateral_accel_g, 8.0)
        self.assertGreater(len(result.speed_mps), 100)


if __name__ == "__main__":
    unittest.main()
