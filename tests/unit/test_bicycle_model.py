"""Unit tests for the solver-facing bicycle model."""

from __future__ import annotations

import unittest

import numpy as np

from lap_time_sim.tire.models import default_axle_tire_parameters
from lap_time_sim.utils.exceptions import ConfigurationError
from lap_time_sim.vehicle import (
    BicycleModel,
    BicycleNumerics,
    BicyclePhysics,
    build_bicycle_model,
)
from lap_time_sim.vehicle.params import default_vehicle_parameters


def _build_bicycle_model() -> BicycleModel:
    """Build a representative solver-facing bicycle model for tests.

    Returns:
        Configured bicycle model instance with explicit physical and numerical
        settings.
    """
    return BicycleModel(
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


class BicycleModelTests(unittest.TestCase):
    """Validate the model API implementation for the bicycle backend."""

    def test_physics_validation_rejects_nonpositive_limits(self) -> None:
        """Reject non-positive configured drive and brake limits."""
        with self.assertRaises(ConfigurationError):
            BicyclePhysics(
                max_drive_accel_mps2=0.0,
                max_brake_accel_mps2=16.0,
                peak_slip_angle_rad=0.12,
            ).validate()
        with self.assertRaises(ConfigurationError):
            BicyclePhysics(
                max_drive_accel_mps2=8.0,
                max_brake_accel_mps2=0.0,
                peak_slip_angle_rad=0.12,
            ).validate()
        with self.assertRaises(ConfigurationError):
            BicyclePhysics(
                max_drive_accel_mps2=8.0,
                max_brake_accel_mps2=16.0,
                peak_slip_angle_rad=0.0,
            ).validate()

    def test_numerics_validation_rejects_invalid_values(self) -> None:
        """Reject invalid numerical settings for lateral limit iteration."""
        with self.assertRaises(ConfigurationError):
            BicycleNumerics(
                min_lateral_accel_limit_mps2=0.0,
                lateral_limit_max_iterations=12,
                lateral_limit_convergence_tol_mps2=0.05,
            ).validate()
        with self.assertRaises(ConfigurationError):
            BicycleNumerics(
                min_lateral_accel_limit_mps2=0.5,
                lateral_limit_max_iterations=0,
                lateral_limit_convergence_tol_mps2=0.05,
            ).validate()
        with self.assertRaises(ConfigurationError):
            BicycleNumerics(
                min_lateral_accel_limit_mps2=0.5,
                lateral_limit_max_iterations=12,
                lateral_limit_convergence_tol_mps2=0.0,
            ).validate()

    def test_diagnostics_are_finite(self) -> None:
        """Return finite diagnostic signals for a representative operating point."""
        model = _build_bicycle_model()
        diagnostics = model.diagnostics(
            speed_mps=45.0,
            ax_mps2=1.2,
            ay_mps2=12.0,
            curvature_1pm=0.03,
        )

        self.assertTrue(np.isfinite(diagnostics.yaw_moment_nm))
        self.assertTrue(np.isfinite(diagnostics.front_axle_load_n))
        self.assertTrue(np.isfinite(diagnostics.rear_axle_load_n))
        self.assertTrue(np.isfinite(diagnostics.power_w))

    def test_uphill_reduces_available_acceleration(self) -> None:
        """Reduce available forward acceleration on positive grade."""
        model = _build_bicycle_model()
        ay_required = 0.0
        on_flat = model.max_longitudinal_accel(
            speed_mps=50.0,
            ay_required_mps2=ay_required,
            grade=0.0,
            banking_rad=0.0,
        )
        uphill = model.max_longitudinal_accel(
            speed_mps=50.0,
            ay_required_mps2=ay_required,
            grade=0.05,
            banking_rad=0.0,
        )

        self.assertLess(uphill, on_flat)

    def test_friction_circle_returns_zero_when_limit_degenerates(self) -> None:
        """Return zero longitudinal capacity when lateral limit is degenerate."""
        model = _build_bicycle_model()
        self.assertEqual(model._friction_circle_scale(ay_required_mps2=5.0, ay_limit_mps2=0.0), 0.0)

    def test_build_bicycle_model_uses_default_numerics(self) -> None:
        """Build a model with default numerical controls when omitted."""
        model = build_bicycle_model(
            vehicle=default_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=BicyclePhysics(
                max_drive_accel_mps2=8.0,
                max_brake_accel_mps2=16.0,
                peak_slip_angle_rad=0.12,
            ),
        )
        self.assertEqual(model.numerics, BicycleNumerics())


if __name__ == "__main__":
    unittest.main()
