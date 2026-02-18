"""Unit tests for the solver-facing single_track model."""

from __future__ import annotations

import unittest

import numpy as np

from apexsim.tire.models import default_axle_tire_parameters
from apexsim.utils.exceptions import ConfigurationError
from apexsim.vehicle import (
    SingleTrackModel,
    SingleTrackNumerics,
    SingleTrackPhysics,
    build_single_track_model,
)
from tests.helpers import sample_vehicle_parameters


def _build_single_track_model() -> SingleTrackModel:
    """Build a representative solver-facing single_track model for tests.

    Returns:
        Configured single_track model instance with explicit physical and numerical
        settings.
    """
    return SingleTrackModel(
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


class SingleTrackModelTests(unittest.TestCase):
    """Validate the model API implementation for the single_track backend."""

    def test_physics_validation_rejects_nonpositive_limits(self) -> None:
        """Reject non-positive configured drive and brake limits."""
        with self.assertRaises(ConfigurationError):
            SingleTrackPhysics(
                max_drive_accel=0.0,
                max_brake_accel=16.0,
                peak_slip_angle=0.12,
            ).validate()
        with self.assertRaises(ConfigurationError):
            SingleTrackPhysics(
                max_drive_accel=8.0,
                max_brake_accel=0.0,
                peak_slip_angle=0.12,
            ).validate()
        with self.assertRaises(ConfigurationError):
            SingleTrackPhysics(
                max_drive_accel=8.0,
                max_brake_accel=16.0,
                peak_slip_angle=0.0,
            ).validate()

    def test_numerics_validation_rejects_invalid_values(self) -> None:
        """Reject invalid numerical settings for lateral limit iteration."""
        with self.assertRaises(ConfigurationError):
            SingleTrackNumerics(
                min_lateral_accel_limit=0.0,
                lateral_limit_max_iterations=12,
                lateral_limit_convergence_tolerance=0.05,
            ).validate()
        with self.assertRaises(ConfigurationError):
            SingleTrackNumerics(
                min_lateral_accel_limit=0.5,
                lateral_limit_max_iterations=0,
                lateral_limit_convergence_tolerance=0.05,
            ).validate()
        with self.assertRaises(ConfigurationError):
            SingleTrackNumerics(
                min_lateral_accel_limit=0.5,
                lateral_limit_max_iterations=12,
                lateral_limit_convergence_tolerance=0.0,
            ).validate()

    def test_diagnostics_are_finite(self) -> None:
        """Return finite diagnostic signals for a representative operating point."""
        model = _build_single_track_model()
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
        model = _build_single_track_model()
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

    def test_lateral_limit_batch_matches_scalar_api(self) -> None:
        """Match vectorized lateral-limit output against scalar API evaluation."""
        model = _build_single_track_model()
        speed = np.array([12.0, 37.0, 68.0], dtype=float)
        banking = np.array([0.01, 0.03, -0.02], dtype=float)

        batch = model.lateral_accel_limit_batch(speed=speed, banking=banking)
        scalar = np.array(
            [
                model.lateral_accel_limit(speed=float(speed[idx]), banking=float(banking[idx]))
                for idx in range(speed.size)
            ],
            dtype=float,
        )
        np.testing.assert_allclose(batch, scalar, rtol=1e-12, atol=1e-12)

    def test_diagnostics_batch_matches_scalar_api(self) -> None:
        """Match vectorized diagnostics output against scalar API evaluation."""
        model = _build_single_track_model()
        speed = np.array([25.0, 44.0, 61.0], dtype=float)
        longitudinal_accel = np.array([1.0, 0.2, -1.4], dtype=float)
        lateral_accel = np.array([6.0, 9.0, 7.5], dtype=float)
        curvature = np.array([0.01, 0.03, -0.02], dtype=float)

        batch = model.diagnostics_batch(
            speed=speed,
            longitudinal_accel=longitudinal_accel,
            lateral_accel=lateral_accel,
            curvature=curvature,
        )
        scalar_yaw = np.zeros_like(speed)
        scalar_front = np.zeros_like(speed)
        scalar_rear = np.zeros_like(speed)
        scalar_power = np.zeros_like(speed)
        for idx in range(speed.size):
            diagnostics = model.diagnostics(
                speed=float(speed[idx]),
                longitudinal_accel=float(longitudinal_accel[idx]),
                lateral_accel=float(lateral_accel[idx]),
                curvature=float(curvature[idx]),
            )
            scalar_yaw[idx] = diagnostics.yaw_moment
            scalar_front[idx] = diagnostics.front_axle_load
            scalar_rear[idx] = diagnostics.rear_axle_load
            scalar_power[idx] = diagnostics.power

        np.testing.assert_allclose(batch[0], scalar_yaw, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(batch[1], scalar_front, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(batch[2], scalar_rear, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(batch[3], scalar_power, rtol=1e-12, atol=1e-12)

    def test_friction_circle_returns_zero_when_limit_degenerates(self) -> None:
        """Return zero longitudinal capacity when lateral limit is degenerate."""
        model = _build_single_track_model()
        self.assertEqual(
            model._friction_circle_scale(lateral_accel_required=5.0, lateral_accel_limit=0.0),
            0.0,
        )

    def test_build_single_track_model_uses_default_numerics(self) -> None:
        """Build a model with default numerical controls when omitted."""
        model = build_single_track_model(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(
                max_drive_accel=8.0,
                max_brake_accel=16.0,
                peak_slip_angle=0.12,
            ),
        )
        self.assertEqual(model.numerics, SingleTrackNumerics())


if __name__ == "__main__":
    unittest.main()
