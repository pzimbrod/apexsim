"""Integration tests for end-to-end lap simulation."""

from __future__ import annotations

import unittest
from pathlib import Path

from lap_time_sim.analysis.kpi import compute_kpis
from lap_time_sim.simulation.config import SimulationConfig
from lap_time_sim.simulation.runner import simulate_lap
from lap_time_sim.tire.models import default_axle_tire_parameters
from lap_time_sim.track.io import load_track_csv
from lap_time_sim.vehicle import BicycleLapTimeModel
from lap_time_sim.vehicle.params import default_vehicle_parameters


class SimulationPipelineTests(unittest.TestCase):
    """End-to-end system tests."""

    def test_full_spa_simulation_runs_and_returns_valid_kpis(self) -> None:
        root = Path(__file__).resolve().parents[2]
        track = load_track_csv(root / "data" / "spa_francorchamps.csv")
        model = BicycleLapTimeModel(
            vehicle=default_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
        )

        result = simulate_lap(
            track=track,
            model=model,
            config=SimulationConfig(),
        )
        kpis = compute_kpis(result)

        self.assertGreater(kpis.lap_time_s, 70.0)
        self.assertLess(kpis.lap_time_s, 220.0)
        self.assertGreater(kpis.max_lateral_accel_g, 1.5)
        self.assertLess(kpis.max_lateral_accel_g, 8.0)
        self.assertGreater(len(result.speed_mps), 100)


if __name__ == "__main__":
    unittest.main()
