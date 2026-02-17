"""Unit tests for the solver-facing point-mass model."""

from __future__ import annotations

import unittest

import numpy as np

from lap_time_sim.tire.models import default_axle_tire_parameters
from lap_time_sim.utils.exceptions import ConfigurationError
from lap_time_sim.vehicle import (
    PointMassModel,
    PointMassPhysics,
    build_point_mass_model,
    calibrate_point_mass_friction_to_bicycle,
)
from tests.helpers import sample_vehicle_parameters


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

    def test_lateral_limit_increases_with_speed_due_to_downforce(self) -> None:
        """Increase lateral limit with speed when aerodynamic downforce is positive."""
        model = _build_point_mass_model()
        low_speed = model.lateral_accel_limit(speed=20.0, banking=0.0)
        high_speed = model.lateral_accel_limit(speed=80.0, banking=0.0)
        self.assertGreater(high_speed, low_speed)

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

    def test_build_point_mass_model_uses_default_physics(self) -> None:
        """Build a model with default physical settings when omitted."""
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        self.assertEqual(model.physics, PointMassPhysics())

    def test_calibration_returns_reasonable_positive_friction(self) -> None:
        """Return a physically plausible positive friction fit from bicycle limits."""
        calibration = calibrate_point_mass_friction_to_bicycle(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
        )
        self.assertGreater(calibration.friction_coefficient, 0.5)
        self.assertLess(calibration.friction_coefficient, 2.5)
        self.assertEqual(calibration.speed_samples.size, calibration.mu_samples.size)

    def test_calibration_rejects_invalid_speed_samples(self) -> None:
        """Reject empty and non-positive speed sample arrays."""
        with self.assertRaises(ConfigurationError):
            calibrate_point_mass_friction_to_bicycle(
                vehicle=sample_vehicle_parameters(),
                tires=default_axle_tire_parameters(),
                speed_samples=np.array([], dtype=float),
            )
        with self.assertRaises(ConfigurationError):
            calibrate_point_mass_friction_to_bicycle(
                vehicle=sample_vehicle_parameters(),
                tires=default_axle_tire_parameters(),
                speed_samples=np.array([10.0, 0.0, 20.0], dtype=float),
            )


if __name__ == "__main__":
    unittest.main()
