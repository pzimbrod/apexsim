"""Tests for lateral envelope convergence behavior."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from apexsim.simulation.config import NumericsConfig, RuntimeConfig, SimulationConfig
from apexsim.simulation.numba_profile import solve_speed_profile_numba
from apexsim.simulation.profile import solve_speed_profile
from apexsim.simulation.torch_profile import solve_speed_profile_torch
from apexsim.tire.models import default_axle_tire_parameters
from apexsim.track.io import load_track_csv
from apexsim.utils.exceptions import ConfigurationError
from apexsim.vehicle import SingleTrackModel, SingleTrackNumerics, SingleTrackPhysics
from tests.helpers import sample_vehicle_parameters

NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class SpeedProfileConvergenceTests(unittest.TestCase):
    """Validate configurable early stopping of the lateral envelope loop."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load shared track/model fixtures for convergence tests."""
        root = Path(__file__).resolve().parents[2]
        cls.track = load_track_csv(root / "data" / "spa_francorchamps.csv")
        cls.model = SingleTrackModel(
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
            ),
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
            ),
        )
        result = solve_speed_profile(self.track, self.model, config)
        self.assertGreater(result.lateral_envelope_iterations, 1)
        self.assertLessEqual(
            result.lateral_envelope_iterations,
            config.numerics.lateral_envelope_max_iterations,
        )

    def test_torch_profile_rejects_non_torch_backend(self) -> None:
        """Reject torch-profile calls when backend is not set to torch."""
        config = SimulationConfig(
            runtime=RuntimeConfig(
                max_speed=115.0,
                enable_transient_refinement=False,
                compute_backend="numpy",
            ),
            numerics=NumericsConfig(),
        )
        with self.assertRaises(ConfigurationError):
            solve_speed_profile_torch(track=self.track, model=self.model, config=config)

    def test_numba_profile_rejects_non_numba_backend(self) -> None:
        """Reject numba-profile calls when backend is not set to numba."""
        config = SimulationConfig(
            runtime=RuntimeConfig(
                max_speed=115.0,
                enable_transient_refinement=False,
                compute_backend="numpy",
            ),
            numerics=NumericsConfig(),
        )
        with self.assertRaises(ConfigurationError):
            solve_speed_profile_numba(track=self.track, model=self.model, config=config)

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba not installed")
    def test_numba_profile_accepts_single_track_model_backend_api(self) -> None:
        """Run numba profile solve successfully with single_track backend adapters."""
        config = SimulationConfig(
            runtime=RuntimeConfig(
                max_speed=115.0,
                enable_transient_refinement=False,
                compute_backend="numba",
            ),
            numerics=NumericsConfig(),
        )
        result = solve_speed_profile_numba(track=self.track, model=self.model, config=config)
        self.assertEqual(result.speed.shape, self.track.arc_length.shape)
        self.assertGreater(result.lap_time, 0.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_torch_profile_accepts_single_track_model_backend_api(self) -> None:
        """Run torch profile solve successfully with single_track backend adapters."""
        config = SimulationConfig(
            runtime=RuntimeConfig(
                max_speed=115.0,
                enable_transient_refinement=False,
                compute_backend="torch",
                torch_device="cpu",
            ),
            numerics=NumericsConfig(),
        )
        result = solve_speed_profile_torch(track=self.track, model=self.model, config=config)
        self.assertEqual(result.speed.shape, self.track.arc_length.shape)
        self.assertGreater(result.lap_time, 0.0)


if __name__ == "__main__":
    unittest.main()
