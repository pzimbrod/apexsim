"""Unit tests for the bicycle lap-time model adapter."""

from __future__ import annotations

import unittest

import numpy as np

from lap_time_sim.tire.models import default_axle_tire_parameters
from lap_time_sim.utils.exceptions import ConfigurationError
from lap_time_sim.vehicle import BicycleLapTimeModel, BicycleLapTimeModelConfig
from lap_time_sim.vehicle.params import default_vehicle_parameters


class BicycleLapTimeModelTests(unittest.TestCase):
    """Validate the model API implementation for the bicycle backend."""

    def test_config_validation_rejects_nonpositive_limits(self) -> None:
        with self.assertRaises(ConfigurationError):
            BicycleLapTimeModelConfig(max_drive_accel_mps2=0.0).validate()
        with self.assertRaises(ConfigurationError):
            BicycleLapTimeModelConfig(max_brake_accel_mps2=0.0).validate()

    def test_diagnostics_are_finite(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
