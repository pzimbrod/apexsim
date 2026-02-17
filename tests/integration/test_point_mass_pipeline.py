"""Integration tests for end-to-end lap simulation with point-mass model."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from pylapsim.analysis.kpi import compute_kpis
from pylapsim.simulation import build_simulation_config, simulate_lap
from pylapsim.track.io import load_track_csv
from pylapsim.vehicle import PointMassPhysics, build_point_mass_model
from tests.helpers import sample_vehicle_parameters

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None


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

        self.assertGreater(kpis.lap_time, 70.0)
        self.assertLess(kpis.lap_time, 260.0)
        self.assertGreater(kpis.max_lateral_accel_g, 1.2)
        self.assertLess(kpis.max_lateral_accel_g, 8.0)
        self.assertGreater(len(result.speed), 100)

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba not installed")
    def test_numba_backend_matches_numpy_backend_for_point_mass(self) -> None:
        """Match numba and NumPy solver outputs for point-mass backend."""
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
    def test_torch_backend_matches_numpy_backend_for_point_mass(self) -> None:
        """Match torch and NumPy solver outputs for point-mass backend."""
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

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_torch_compile_matches_plain_torch_backend(self) -> None:
        """Match compiled and non-compiled torch backend outputs."""
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

        plain_torch = simulate_lap(
            track=track,
            model=model,
            config=build_simulation_config(compute_backend="torch", torch_device="cpu"),
        )
        compiled_torch = simulate_lap(
            track=track,
            model=model,
            config=build_simulation_config(
                compute_backend="torch",
                torch_device="cpu",
                torch_compile=True,
            ),
        )

        np.testing.assert_allclose(
            compiled_torch.speed,
            plain_torch.speed,
            rtol=1e-9,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            compiled_torch.longitudinal_accel,
            plain_torch.longitudinal_accel,
            rtol=1e-8,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            compiled_torch.lateral_accel,
            plain_torch.lateral_accel,
            rtol=1e-9,
            atol=1e-9,
        )
        self.assertAlmostEqual(compiled_torch.lap_time, plain_torch.lap_time, places=9)


if __name__ == "__main__":
    unittest.main()
