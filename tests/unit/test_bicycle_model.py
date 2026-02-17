"""Unit tests for the solver-facing bicycle model."""

from __future__ import annotations

import unittest

import numpy as np

from pylapsim.tire.models import default_axle_tire_parameters
from pylapsim.utils.exceptions import ConfigurationError
from pylapsim.vehicle import (
    BicycleModel,
    BicycleNumerics,
    BicyclePhysics,
    build_bicycle_model,
)
from tests.helpers import sample_vehicle_parameters


def _build_bicycle_model() -> BicycleModel:
    """Build a representative solver-facing bicycle model for tests.

    Returns:
        Configured bicycle model instance with explicit physical and numerical
        settings.
    """
    return BicycleModel(
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


class BicycleModelTests(unittest.TestCase):
    """Validate the model API implementation for the bicycle backend."""

    def test_physics_validation_rejects_nonpositive_limits(self) -> None:
        """Reject non-positive configured drive and brake limits."""
        with self.assertRaises(ConfigurationError):
            BicyclePhysics(
                max_drive_accel=0.0,
                max_brake_accel=16.0,
                peak_slip_angle=0.12,
            ).validate()
        with self.assertRaises(ConfigurationError):
            BicyclePhysics(
                max_drive_accel=8.0,
                max_brake_accel=0.0,
                peak_slip_angle=0.12,
            ).validate()
        with self.assertRaises(ConfigurationError):
            BicyclePhysics(
                max_drive_accel=8.0,
                max_brake_accel=16.0,
                peak_slip_angle=0.0,
            ).validate()

    def test_numerics_validation_rejects_invalid_values(self) -> None:
        """Reject invalid numerical settings for lateral limit iteration."""
        with self.assertRaises(ConfigurationError):
            BicycleNumerics(
                min_lateral_accel_limit=0.0,
                lateral_limit_max_iterations=12,
                lateral_limit_convergence_tolerance=0.05,
            ).validate()
        with self.assertRaises(ConfigurationError):
            BicycleNumerics(
                min_lateral_accel_limit=0.5,
                lateral_limit_max_iterations=0,
                lateral_limit_convergence_tolerance=0.05,
            ).validate()
        with self.assertRaises(ConfigurationError):
            BicycleNumerics(
                min_lateral_accel_limit=0.5,
                lateral_limit_max_iterations=12,
                lateral_limit_convergence_tolerance=0.0,
            ).validate()

    def test_diagnostics_are_finite(self) -> None:
        """Return finite diagnostic signals for a representative operating point."""
        model = _build_bicycle_model()
        diagnostics = model.diagnostics(
            speed=45.0,
            longitudinal_accel=1.2,
            lateral_accel=12.0,
            curvature=0.03,
        )

        self.assertTrue(np.isfinite(diagnostics.yaw_moment))
        self.assertTrue(np.isfinite(diagnostics.front_axle_load))
        self.assertTrue(np.isfinite(diagnostics.rear_axle_load))
        self.assertTrue(np.isfinite(diagnostics.power))

    def test_uphill_reduces_available_acceleration(self) -> None:
        """Reduce available forward acceleration on positive grade."""
        model = _build_bicycle_model()
        ay_required = 0.0
        on_flat = model.max_longitudinal_accel(
            speed=50.0,
            lateral_accel_required=ay_required,
            grade=0.0,
            banking=0.0,
        )
        uphill = model.max_longitudinal_accel(
            speed=50.0,
            lateral_accel_required=ay_required,
            grade=0.05,
            banking=0.0,
        )

        self.assertLess(uphill, on_flat)

    def test_friction_circle_returns_zero_when_limit_degenerates(self) -> None:
        """Return zero longitudinal capacity when lateral limit is degenerate."""
        model = _build_bicycle_model()
        self.assertEqual(
            model._friction_circle_scale(lateral_accel_required=5.0, lateral_accel_limit=0.0),
            0.0,
        )

    def test_build_bicycle_model_uses_default_numerics(self) -> None:
        """Build a model with default numerical controls when omitted."""
        model = build_bicycle_model(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=BicyclePhysics(
                max_drive_accel=8.0,
                max_brake_accel=16.0,
                peak_slip_angle=0.12,
            ),
        )
        self.assertEqual(model.numerics, BicycleNumerics())


if __name__ == "__main__":
    unittest.main()
