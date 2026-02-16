"""Unit tests for vehicle dynamics helpers."""

from __future__ import annotations

import unittest

from lap_time_sim.tire import default_axle_tire_parameters
from lap_time_sim.utils.constants import GRAVITY_MPS2
from lap_time_sim.vehicle.aero import aero_forces
from lap_time_sim.vehicle.bicycle import BicycleModel, ControlInput, VehicleState
from lap_time_sim.vehicle.load_transfer import estimate_normal_loads
from lap_time_sim.vehicle.params import default_vehicle_parameters


class VehicleDynamicsTests(unittest.TestCase):
    """Vehicle-level dynamics checks."""

    def test_drag_scales_with_speed_square(self) -> None:
        """Scale aerodynamic drag approximately with speed squared."""
        vehicle = default_vehicle_parameters()
        drag_50 = aero_forces(vehicle, 50.0).drag_n
        drag_100 = aero_forces(vehicle, 100.0).drag_n
        self.assertAlmostEqual(drag_100 / drag_50, 4.0, delta=0.12)

    def test_normal_load_total_matches_weight_plus_downforce(self) -> None:
        """Preserve total vertical load as weight plus downforce."""
        vehicle = default_vehicle_parameters()
        loads = estimate_normal_loads(
            vehicle,
            speed_mps=60.0,
            longitudinal_accel_mps2=0.0,
            lateral_accel_mps2=0.0,
        )
        aero = aero_forces(vehicle, 60.0)
        total = loads.front_axle_n + loads.rear_axle_n
        expected = vehicle.mass_kg * GRAVITY_MPS2 + aero.downforce_n
        self.assertAlmostEqual(total, expected, delta=1e-6)

    def test_bicycle_derivatives_are_finite(self) -> None:
        """Return finite derivatives for a representative dynamic state."""
        vehicle = default_vehicle_parameters()
        model = BicycleModel(vehicle, default_axle_tire_parameters())
        state = VehicleState(vx_mps=40.0, vy_mps=0.7, yaw_rate_rps=0.12)
        control = ControlInput(steer_rad=0.05, longitudinal_accel_cmd_mps2=1.5)

        derivatives = model.derivatives(state, control)
        self.assertTrue(abs(derivatives.vx_mps) < 100.0)
        self.assertTrue(abs(derivatives.vy_mps) < 100.0)
        self.assertTrue(abs(derivatives.yaw_rate_rps) < 100.0)

    def test_wheel_loads_preserve_axle_load_under_high_lateral_accel(self) -> None:
        """Preserve axle totals and positive wheel loads under high lateral acceleration."""
        vehicle = default_vehicle_parameters()
        loads = estimate_normal_loads(
            vehicle=vehicle,
            speed_mps=70.0,
            longitudinal_accel_mps2=0.0,
            lateral_accel_mps2=4.0 * GRAVITY_MPS2,
        )

        self.assertAlmostEqual(
            loads.front_left_n + loads.front_right_n,
            loads.front_axle_n,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            loads.rear_left_n + loads.rear_right_n,
            loads.rear_axle_n,
            delta=1e-6,
        )
        self.assertGreater(loads.front_left_n, 0.0)
        self.assertGreater(loads.front_right_n, 0.0)
        self.assertGreater(loads.rear_left_n, 0.0)
        self.assertGreater(loads.rear_right_n, 0.0)

    def test_state_array_roundtrip_and_speed_sanitization(self) -> None:
        """Round-trip state conversion helpers and sanitize very low speeds."""
        state = VehicleState(vx_mps=15.0, vy_mps=-0.4, yaw_rate_rps=0.2)
        values = BicycleModel.to_array(state)
        restored = BicycleModel.from_array(values)
        self.assertEqual(restored, state)
        self.assertGreater(BicycleModel.sanitize_speed(0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
