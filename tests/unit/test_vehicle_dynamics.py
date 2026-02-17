"""Unit tests for vehicle dynamics helpers."""

from __future__ import annotations

import unittest

from pylapsim.tire import default_axle_tire_parameters
from pylapsim.utils.constants import GRAVITY
from pylapsim.vehicle.aero import aero_forces
from pylapsim.vehicle.bicycle.dynamics import BicycleDynamicsModel, ControlInput, VehicleState
from pylapsim.vehicle.bicycle.load_transfer import estimate_normal_loads
from tests.helpers import sample_vehicle_parameters


class VehicleDynamicsTests(unittest.TestCase):
    """Vehicle-level dynamics checks."""

    def test_drag_scales_with_speed_square(self) -> None:
        """Scale aerodynamic drag approximately with speed squared."""
        vehicle = sample_vehicle_parameters()
        drag_50 = aero_forces(vehicle, 50.0).drag
        drag_100 = aero_forces(vehicle, 100.0).drag
        self.assertAlmostEqual(drag_100 / drag_50, 4.0, delta=0.12)

    def test_normal_load_total_matches_weight_plus_downforce(self) -> None:
        """Preserve total vertical load as weight plus downforce."""
        vehicle = sample_vehicle_parameters()
        loads = estimate_normal_loads(
            vehicle,
            speed=60.0,
            longitudinal_accel=0.0,
            lateral_accel=0.0,
        )
        aero = aero_forces(vehicle, 60.0)
        total = loads.front_axle_load + loads.rear_axle_load
        expected = vehicle.mass * GRAVITY + aero.downforce
        self.assertAlmostEqual(total, expected, delta=1e-6)

    def test_bicycle_derivatives_are_finite(self) -> None:
        """Return finite derivatives for a representative dynamic state."""
        vehicle = sample_vehicle_parameters()
        model = BicycleDynamicsModel(vehicle, default_axle_tire_parameters())
        state = VehicleState(vx=40.0, vy=0.7, yaw_rate=0.12)
        control = ControlInput(steer=0.05, longitudinal_accel_cmd=1.5)

        derivatives = model.derivatives(state, control)
        self.assertTrue(abs(derivatives.vx) < 100.0)
        self.assertTrue(abs(derivatives.vy) < 100.0)
        self.assertTrue(abs(derivatives.yaw_rate) < 100.0)

    def test_wheel_loads_preserve_axle_load_under_high_lateral_accel(self) -> None:
        """Preserve axle totals and positive wheel loads under high lateral acceleration."""
        vehicle = sample_vehicle_parameters()
        loads = estimate_normal_loads(
            vehicle=vehicle,
            speed=70.0,
            longitudinal_accel=0.0,
            lateral_accel=4.0 * GRAVITY,
        )

        self.assertAlmostEqual(
            loads.front_left_load + loads.front_right_load,
            loads.front_axle_load,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            loads.rear_left_load + loads.rear_right_load,
            loads.rear_axle_load,
            delta=1e-6,
        )
        self.assertGreater(loads.front_left_load, 0.0)
        self.assertGreater(loads.front_right_load, 0.0)
        self.assertGreater(loads.rear_left_load, 0.0)
        self.assertGreater(loads.rear_right_load, 0.0)

    def test_state_array_roundtrip_and_speed_sanitization(self) -> None:
        """Round-trip state conversion helpers and sanitize very low speeds."""
        state = VehicleState(vx=15.0, vy=-0.4, yaw_rate=0.2)
        values = BicycleDynamicsModel.to_array(state)
        restored = BicycleDynamicsModel.from_array(values)
        self.assertEqual(restored, state)
        self.assertGreater(BicycleDynamicsModel.sanitize_speed(0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
