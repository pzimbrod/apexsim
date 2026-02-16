"""Integration tests for end-to-end lap simulation."""

from __future__ import annotations

import unittest
from pathlib import Path

from lap_time_sim.analysis.kpi import compute_kpis
from lap_time_sim.simulation.config import NumericsConfig, RuntimeConfig, SimulationConfig
from lap_time_sim.simulation.runner import simulate_lap
from lap_time_sim.tire.models import default_axle_tire_parameters
from lap_time_sim.track.io import load_track_csv
from lap_time_sim.vehicle import BicycleModel, BicycleNumerics, BicyclePhysics
from lap_time_sim.vehicle.params import default_vehicle_parameters


class SimulationPipelineTests(unittest.TestCase):
    """End-to-end system tests."""

    def test_full_spa_simulation_runs_and_returns_valid_kpis(self) -> None:
        """Verify that the full Spa pipeline returns plausible KPI ranges."""
        root = Path(__file__).resolve().parents[2]
        track = load_track_csv(root / "data" / "spa_francorchamps.csv")
        model = BicycleModel(
            vehicle=default_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=BicyclePhysics(
                max_drive_accel_mps2=8.0,
                max_brake_accel_mps2=16.0,
                peak_slip_angle_rad=0.12,
            ),
            numerics=BicycleNumerics(
                min_lateral_accel_limit_mps2=0.5,
                lateral_limit_max_iterations=12,
                lateral_limit_convergence_tol_mps2=0.05,
            ),
        )

        result = simulate_lap(
            track=track,
            model=model,
            config=SimulationConfig(
                runtime=RuntimeConfig(
                    max_speed_mps=115.0,
                    enable_transient_refinement=False,
                ),
                numerics=NumericsConfig(
                    min_speed_mps=8.0,
                    lateral_envelope_max_iterations=20,
                    lateral_envelope_convergence_tol_mps=0.1,
                    transient_dt_s=0.01,
                ),
            ),
        )
        kpis = compute_kpis(result)

        self.assertGreater(kpis.lap_time_s, 70.0)
        self.assertLess(kpis.lap_time_s, 220.0)
        self.assertGreater(kpis.max_lateral_accel_g, 1.5)
        self.assertLess(kpis.max_lateral_accel_g, 8.0)
        self.assertGreater(len(result.speed_mps), 100)


if __name__ == "__main__":
    unittest.main()
