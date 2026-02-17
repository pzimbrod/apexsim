"""Unit tests for the solver-facing point-mass model."""

from __future__ import annotations

import unittest

import numpy as np

from lap_time_sim.utils.exceptions import ConfigurationError
from lap_time_sim.vehicle import PointMassModel, PointMassPhysics, build_point_mass_model
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
        low_speed = model.lateral_accel_limit(speed_mps=20.0, banking_rad=0.0)
        high_speed = model.lateral_accel_limit(speed_mps=80.0, banking_rad=0.0)
        self.assertGreater(high_speed, low_speed)

    def test_uphill_reduces_available_acceleration(self) -> None:
        """Reduce available forward acceleration on positive grade."""
        model = _build_point_mass_model()
        on_flat = model.max_longitudinal_accel(
            speed_mps=50.0,
            ay_required_mps2=0.0,
            grade=0.0,
            banking_rad=0.0,
        )
        uphill = model.max_longitudinal_accel(
            speed_mps=50.0,
            ay_required_mps2=0.0,
            grade=0.05,
            banking_rad=0.0,
        )
        self.assertLess(uphill, on_flat)

    def test_diagnostics_are_finite_and_yaw_moment_is_zero(self) -> None:
        """Return finite diagnostic signals and zero yaw moment by model definition."""
        model = _build_point_mass_model()
        diagnostics = model.diagnostics(
            speed_mps=45.0,
            ax_mps2=1.2,
            ay_mps2=10.0,
            curvature_1pm=0.03,
        )

        self.assertEqual(diagnostics.yaw_moment_nm, 0.0)
        self.assertTrue(np.isfinite(diagnostics.front_axle_load_n))
        self.assertTrue(np.isfinite(diagnostics.rear_axle_load_n))
        self.assertTrue(np.isfinite(diagnostics.power_w))

    def test_friction_circle_returns_zero_when_limit_degenerates(self) -> None:
        """Return zero longitudinal capacity when lateral limit is degenerate."""
        model = _build_point_mass_model()
        self.assertEqual(model._friction_circle_scale(ay_required_mps2=5.0, ay_limit_mps2=0.0), 0.0)

    def test_build_point_mass_model_uses_default_physics(self) -> None:
        """Build a model with default physical settings when omitted."""
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        self.assertEqual(model.physics, PointMassPhysics())


if __name__ == "__main__":
    unittest.main()
