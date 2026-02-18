"""Integration tests for end-to-end lap simulation."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from apexsim.analysis.kpi import compute_kpis
from apexsim.simulation.config import (
    NumericsConfig,
    RuntimeConfig,
    SimulationConfig,
    build_simulation_config,
)
from apexsim.simulation.runner import simulate_lap
from apexsim.tire.models import default_axle_tire_parameters
from apexsim.track.io import load_track_csv
from apexsim.vehicle import SingleTrackModel, SingleTrackNumerics, SingleTrackPhysics
from tests.helpers import sample_vehicle_parameters

NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class SimulationPipelineTests(unittest.TestCase):
    """End-to-end system tests."""

    def test_full_spa_simulation_runs_and_returns_valid_kpis(self) -> None:
        """Verify that the full Spa pipeline returns plausible KPI ranges."""
        root = Path(__file__).resolve().parents[2]
        track = load_track_csv(root / "data" / "spa_francorchamps.csv")
        model = SingleTrackModel(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(
                max_drive_accel=8.0,
                max_brake_accel=16.0,
                peak_slip_angle=0.12,
            ),
            numerics=SingleTrackNumerics(
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

        self.assertGreater(kpis.lap_time, 70.0)
        self.assertLess(kpis.lap_time, 220.0)
        self.assertGreater(kpis.max_lateral_accel_g, 1.5)
        self.assertLess(kpis.max_lateral_accel_g, 8.0)
        self.assertGreater(len(result.speed), 100)

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba not installed")
    def test_numba_backend_matches_numpy_backend_for_single_track(self) -> None:
        """Match numba and NumPy solver outputs for single_track backend."""
        root = Path(__file__).resolve().parents[2]
        track = load_track_csv(root / "data" / "spa_francorchamps.csv")
        model = SingleTrackModel(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(),
            numerics=SingleTrackNumerics(),
        )

        numpy_result = simulate_lap(
            track=track,
            model=model,
            config=build_simulation_config(compute_backend="numpy"),
        )
        numba_result = simulate_lap(
            track=track,
            model=model,
            config=build_simulation_config(compute_backend="numba"),
        )

        np.testing.assert_allclose(numba_result.speed, numpy_result.speed, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            numba_result.longitudinal_accel,
            numpy_result.longitudinal_accel,
            rtol=1e-9,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            numba_result.lateral_accel,
            numpy_result.lateral_accel,
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertAlmostEqual(numba_result.lap_time, numpy_result.lap_time, places=10)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_torch_backend_matches_numpy_backend_for_single_track(self) -> None:
        """Match torch and NumPy solver outputs for single_track backend."""
        root = Path(__file__).resolve().parents[2]
        track = load_track_csv(root / "data" / "spa_francorchamps.csv")
        model = SingleTrackModel(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(),
            numerics=SingleTrackNumerics(),
        )

        numpy_result = simulate_lap(
            track=track,
            model=model,
            config=build_simulation_config(compute_backend="numpy"),
        )
        torch_result = simulate_lap(
            track=track,
            model=model,
            config=build_simulation_config(compute_backend="torch", torch_device="cpu"),
        )

        np.testing.assert_allclose(torch_result.speed, numpy_result.speed, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            torch_result.longitudinal_accel,
            numpy_result.longitudinal_accel,
            rtol=1e-8,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            torch_result.lateral_accel,
            numpy_result.lateral_accel,
            rtol=1e-9,
            atol=1e-9,
        )
        self.assertAlmostEqual(torch_result.lap_time, numpy_result.lap_time, places=9)


if __name__ == "__main__":
    unittest.main()
