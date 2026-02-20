"""Unit tests for the solver-facing point-mass model."""

from __future__ import annotations

import importlib.util
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from apexsim.tire.models import default_axle_tire_parameters
from apexsim.utils.exceptions import ConfigurationError
from apexsim.vehicle import (
    PointMassModel,
    PointMassPhysics,
    build_point_mass_model,
    calibrate_point_mass_friction_to_single_track,
)
from apexsim.vehicle.single_track_model import SingleTrackModel
from tests.helpers import sample_vehicle_parameters

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _build_point_mass_model() -> PointMassModel:
    """Build a representative solver-facing point-mass model for tests.

    Returns:
        Configured point-mass model instance with explicit physical settings.
    """
    return PointMassModel(
        vehicle=sample_vehicle_parameters(),
        physics=PointMassPhysics(
            max_drive_accel=8.0,
            max_brake_accel=16.0,
            friction_coefficient=1.7,
        ),
    )


class PointMassModelTests(unittest.TestCase):
    """Validate the model API implementation for the point-mass backend."""

    def test_physics_validation_rejects_nonpositive_limits(self) -> None:
        """Reject non-positive configured drive and brake limits."""
        with self.assertRaises(ConfigurationError):
            PointMassPhysics(
                max_drive_accel=0.0,
                max_brake_accel=16.0,
                friction_coefficient=1.7,
            ).validate()
        with self.assertRaises(ConfigurationError):
            PointMassPhysics(
                max_drive_accel=8.0,
                max_brake_accel=0.0,
                friction_coefficient=1.7,
            ).validate()
        with self.assertRaises(ConfigurationError):
            PointMassPhysics(
                max_drive_accel=8.0,
                max_brake_accel=16.0,
                friction_coefficient=0.0,
            ).validate()
        with self.assertRaises(ConfigurationError):
            PointMassPhysics(
                max_drive_accel=8.0,
                max_brake_accel=16.0,
                reference_mass=0.0,
                friction_coefficient=1.7,
            ).validate()

    def test_reference_mass_scales_longitudinal_limits_with_vehicle_mass(self) -> None:
        """Scale accel/decel limits inversely with mass when reference mass is set."""
        vehicle = sample_vehicle_parameters()
        baseline_mass = vehicle.mass
        physics = PointMassPhysics(
            max_drive_accel=8.0,
            max_brake_accel=16.0,
            reference_mass=baseline_mass,
            friction_coefficient=1.7,
        )
        lighter_model = build_point_mass_model(
            vehicle=replace(vehicle, mass=baseline_mass * 0.9),
            physics=physics,
        )
        heavier_model = build_point_mass_model(
            vehicle=replace(vehicle, mass=baseline_mass * 1.1),
            physics=physics,
        )
        lighter_accel = lighter_model.max_longitudinal_accel(
            speed=20.0,
            lateral_accel_required=0.0,
            grade=0.0,
            banking=0.0,
        )
        heavier_accel = heavier_model.max_longitudinal_accel(
            speed=20.0,
            lateral_accel_required=0.0,
            grade=0.0,
            banking=0.0,
        )
        lighter_brake = lighter_model.max_longitudinal_decel(
            speed=20.0,
            lateral_accel_required=0.0,
            grade=0.0,
            banking=0.0,
        )
        heavier_brake = heavier_model.max_longitudinal_decel(
            speed=20.0,
            lateral_accel_required=0.0,
            grade=0.0,
            banking=0.0,
        )

        self.assertGreater(lighter_accel, heavier_accel)
        self.assertGreater(lighter_brake, heavier_brake)

    def test_lateral_limit_increases_with_speed_due_to_downforce(self) -> None:
        """Increase lateral limit with speed when aerodynamic downforce is positive."""
        model = _build_point_mass_model()
        low_speed = model.lateral_accel_limit(speed=20.0, banking=0.0)
        high_speed = model.lateral_accel_limit(speed=80.0, banking=0.0)
        self.assertGreater(high_speed, low_speed)

    def test_lateral_limit_batch_matches_scalar_api(self) -> None:
        """Match vectorized lateral-limit output against scalar API evaluation."""
        model = _build_point_mass_model()
        speed = np.array([10.0, 30.0, 60.0], dtype=float)
        banking = np.array([0.00, 0.05, -0.02], dtype=float)

        batch = model.lateral_accel_limit_batch(speed=speed, banking=banking)
        scalar = np.array(
            [
                model.lateral_accel_limit(speed=float(speed[idx]), banking=float(banking[idx]))
                for idx in range(speed.size)
            ],
            dtype=float,
        )
        np.testing.assert_allclose(batch, scalar, rtol=1e-12, atol=1e-12)

    def test_uphill_reduces_available_acceleration(self) -> None:
        """Reduce available forward acceleration on positive grade."""
        model = _build_point_mass_model()
        on_flat = model.max_longitudinal_accel(
            speed=50.0,
            lateral_accel_required=0.0,
            grade=0.0,
            banking=0.0,
        )
        uphill = model.max_longitudinal_accel(
            speed=50.0,
            lateral_accel_required=0.0,
            grade=0.05,
            banking=0.0,
        )
        self.assertLess(uphill, on_flat)

    def test_diagnostics_are_finite_and_yaw_moment_is_zero(self) -> None:
        """Return finite diagnostic signals and zero yaw moment by model definition."""
        model = _build_point_mass_model()
        diagnostics = model.diagnostics(
            speed=45.0,
            longitudinal_accel=1.2,
            lateral_accel=10.0,
            curvature=0.03,
        )

        self.assertEqual(diagnostics.yaw_moment, 0.0)
        self.assertTrue(np.isfinite(diagnostics.front_axle_load))
        self.assertTrue(np.isfinite(diagnostics.rear_axle_load))
        self.assertTrue(np.isfinite(diagnostics.power))

    def test_friction_circle_returns_zero_when_limit_degenerates(self) -> None:
        """Return zero longitudinal capacity when lateral limit is degenerate."""
        model = _build_point_mass_model()
        self.assertEqual(
            model._friction_circle_scale(lateral_accel_required=5.0, lateral_accel_limit=0.0),
            0.0,
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_torch_lateral_limit_matches_numpy_and_supports_gradients(self) -> None:
        """Match torch lateral limits to NumPy and expose gradients wrt speed."""
        import torch

        model = _build_point_mass_model()
        speed = torch.tensor([20.0, 45.0, 70.0], dtype=torch.float64, requires_grad=True)
        banking = torch.tensor([0.02, 0.00, -0.01], dtype=torch.float64)

        torch_output = model.lateral_accel_limit_torch(speed=speed, banking=banking)
        numpy_output = model.lateral_accel_limit_batch(
            speed=speed.detach().numpy(),
            banking=banking.detach().numpy(),
        )

        np.testing.assert_allclose(
            torch_output.detach().numpy(),
            numpy_output,
            rtol=1e-10,
            atol=1e-10,
        )

        loss = torch.sum(torch_output)
        loss.backward()
        self.assertIsNotNone(speed.grad)
        self.assertTrue(torch.all(torch.isfinite(speed.grad)))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_torch_longitudinal_limits_match_numpy(self) -> None:
        """Match torch longitudinal-limit outputs to scalar NumPy backend values."""
        import torch

        model = _build_point_mass_model()

        speed = torch.tensor(45.0, dtype=torch.float64)
        lateral_accel_required = torch.tensor(8.0, dtype=torch.float64)
        grade = torch.tensor(0.01, dtype=torch.float64)
        banking = torch.tensor(0.02, dtype=torch.float64)

        torch_accel = model.max_longitudinal_accel_torch(
            speed=speed,
            lateral_accel_required=lateral_accel_required,
            grade=grade,
            banking=banking,
        )
        torch_decel = model.max_longitudinal_decel_torch(
            speed=speed,
            lateral_accel_required=lateral_accel_required,
            grade=grade,
            banking=banking,
        )

        numpy_accel = model.max_longitudinal_accel(
            speed=45.0,
            lateral_accel_required=8.0,
            grade=0.01,
            banking=0.02,
        )
        numpy_decel = model.max_longitudinal_decel(
            speed=45.0,
            lateral_accel_required=8.0,
            grade=0.01,
            banking=0.02,
        )

        self.assertAlmostEqual(float(torch_accel.item()), float(numpy_accel), places=10)
        self.assertAlmostEqual(float(torch_decel.item()), float(numpy_decel), places=10)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_torch_reference_mass_scaling_helpers_cover_both_branches(self) -> None:
        """Evaluate torch reference-mass helpers with and without scaling."""
        import torch

        model_without_reference = _build_point_mass_model()
        drive_unscaled = model_without_reference._backend_scaled_drive_envelope_accel_limit(torch)
        brake_unscaled = model_without_reference._backend_scaled_brake_envelope_accel_limit(torch)
        self.assertAlmostEqual(
            float(drive_unscaled.item()),
            model_without_reference.envelope_physics.max_drive_accel,
            places=12,
        )
        self.assertAlmostEqual(
            float(brake_unscaled.item()),
            model_without_reference.envelope_physics.max_brake_accel,
            places=12,
        )

        vehicle = sample_vehicle_parameters()
        reference_mass = vehicle.mass * 1.1
        model_with_reference = PointMassModel(
            vehicle=vehicle,
            physics=PointMassPhysics(
                max_drive_accel=8.0,
                max_brake_accel=16.0,
                reference_mass=reference_mass,
                friction_coefficient=1.7,
            ),
        )
        drive_scaled = model_with_reference._backend_scaled_drive_envelope_accel_limit(torch)
        brake_scaled = model_with_reference._backend_scaled_brake_envelope_accel_limit(torch)
        self.assertAlmostEqual(
            float(drive_scaled.item()),
            model_with_reference.envelope_physics.max_drive_accel * reference_mass / vehicle.mass,
            places=6,
        )
        self.assertAlmostEqual(
            float(brake_scaled.item()),
            model_with_reference.envelope_physics.max_brake_accel * reference_mass / vehicle.mass,
            places=6,
        )

    def test_build_point_mass_model_uses_default_physics(self) -> None:
        """Build a model with default physical settings when omitted."""
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        self.assertEqual(model.physics, PointMassPhysics())

    def test_calibration_returns_reasonable_positive_friction(self) -> None:
        """Return a physically plausible positive friction fit from single_track limits."""
        calibration = calibrate_point_mass_friction_to_single_track(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
        )
        self.assertGreater(calibration.friction_coefficient, 0.5)
        self.assertLess(calibration.friction_coefficient, 2.5)
        self.assertEqual(calibration.speed_samples.size, calibration.mu_samples.size)

    def test_calibration_rejects_invalid_speed_samples(self) -> None:
        """Reject empty and non-positive speed sample arrays."""
        with self.assertRaises(ConfigurationError):
            calibrate_point_mass_friction_to_single_track(
                vehicle=sample_vehicle_parameters(),
                tires=default_axle_tire_parameters(),
                speed_samples=np.array([], dtype=float),
            )
        with self.assertRaises(ConfigurationError):
            calibrate_point_mass_friction_to_single_track(
                vehicle=sample_vehicle_parameters(),
                tires=default_axle_tire_parameters(),
                speed_samples=np.array([10.0, 0.0, 20.0], dtype=float),
            )

    def test_calibration_uses_single_track_batch_lateral_limit_path(self) -> None:
        """Use vectorized single_track lateral-limit API during calibration."""
        with patch.object(
            SingleTrackModel,
            "lateral_accel_limit",
            side_effect=AssertionError("scalar lateral_accel_limit should not be used"),
        ):
            calibration = calibrate_point_mass_friction_to_single_track(
                vehicle=sample_vehicle_parameters(),
                tires=default_axle_tire_parameters(),
                speed_samples=np.linspace(12.0, 80.0, 9, dtype=float),
            )
        self.assertGreater(calibration.friction_coefficient, 0.0)


if __name__ == "__main__":
    unittest.main()
