"""Unit tests for the bicycle lap-time model adapter."""

from __future__ import annotations

import unittest

import numpy as np

from lap_time_sim.tire.models import default_axle_tire_parameters
from lap_time_sim.utils.exceptions import ConfigurationError
from lap_time_sim.vehicle import (
    BicycleLapTimeModel,
    BicycleLapTimeModelNumerics,
    BicycleLapTimeModelPhysics,
    build_default_bicycle_lap_time_model,
)
from lap_time_sim.vehicle.params import default_vehicle_parameters


class BicycleLapTimeModelTests(unittest.TestCase):
    """Validate the model API implementation for the bicycle backend."""

    def test_physics_validation_rejects_nonpositive_limits(self) -> None:
        """Reject non-positive configured drive and brake limits."""
        with self.assertRaises(ConfigurationError):
            BicycleLapTimeModelPhysics(max_drive_accel_mps2=0.0).validate()
        with self.assertRaises(ConfigurationError):
            BicycleLapTimeModelPhysics(max_brake_accel_mps2=0.0).validate()
        with self.assertRaises(ConfigurationError):
            BicycleLapTimeModelPhysics(peak_slip_angle_rad=0.0).validate()

    def test_numerics_validation_rejects_invalid_values(self) -> None:
        """Reject invalid numerical settings for lateral limit iteration."""
        with self.assertRaises(ConfigurationError):
            BicycleLapTimeModelNumerics(min_lateral_accel_limit_mps2=0.0).validate()
        with self.assertRaises(ConfigurationError):
            BicycleLapTimeModelNumerics(lateral_limit_max_iterations=0).validate()
        with self.assertRaises(ConfigurationError):
            BicycleLapTimeModelNumerics(lateral_limit_convergence_tol_mps2=0.0).validate()

    def test_diagnostics_are_finite(self) -> None:
        """Return finite diagnostic signals for a representative operating point."""
        model = BicycleLapTimeModel(
            vehicle=default_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
        )
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
        model = BicycleLapTimeModel(
            vehicle=default_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
        )
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
        model = build_default_bicycle_lap_time_model()
        self.assertEqual(model._friction_circle_scale(ay_required_mps2=5.0, ay_limit_mps2=0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
