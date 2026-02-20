"""Unit tests for vehicle dynamics helpers."""

from __future__ import annotations

import importlib.util
import unittest
from dataclasses import replace

import numpy as np

from apexsim.tire import default_axle_tire_parameters
from apexsim.utils.constants import GRAVITY
from apexsim.vehicle import SingleTrackNumerics, SingleTrackPhysics, build_single_track_model
from apexsim.vehicle._backend_physics_core import axle_tire_loads_numpy, axle_tire_loads_torch
from apexsim.vehicle.aero import aero_forces
from apexsim.vehicle.single_track.dynamics import (
    ControlInput,
    SingleTrackDynamicsModel,
    VehicleState,
)
from apexsim.vehicle.single_track.load_transfer import estimate_normal_loads
from tests.helpers import sample_vehicle_parameters

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


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

    def test_single_track_derivatives_are_finite(self) -> None:
        """Return finite derivatives for a representative dynamic state."""
        vehicle = sample_vehicle_parameters()
        model = SingleTrackDynamicsModel(vehicle, default_axle_tire_parameters())
        state = VehicleState(vx=40.0, vy=0.7, yaw_rate=0.12)
        control = ControlInput(steer=0.05, longitudinal_accel_cmd=1.5)

        derivatives = model.derivatives(state, control)
        self.assertTrue(abs(derivatives.vx) < 100.0)
        self.assertTrue(abs(derivatives.vy) < 100.0)
        self.assertTrue(abs(derivatives.yaw_rate) < 100.0)

    def test_single_track_longitudinal_command_is_net_acceleration(self) -> None:
        """Interpret longitudinal command directly as net path-tangent acceleration."""
        vehicle = sample_vehicle_parameters()
        model = SingleTrackDynamicsModel(vehicle, default_axle_tire_parameters())
        state = VehicleState(vx=40.0, vy=0.0, yaw_rate=0.0)
        control = ControlInput(steer=0.0, longitudinal_accel_cmd=0.0)

        derivatives = model.derivatives(state, control)
        self.assertAlmostEqual(derivatives.vx, 0.0, delta=1e-12)

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
        values = SingleTrackDynamicsModel.to_array(state)
        restored = SingleTrackDynamicsModel.from_array(values)
        self.assertEqual(restored, state)
        self.assertGreater(SingleTrackDynamicsModel.sanitize_speed(0.0), 0.0)

    def test_single_track_lateral_capacity_decreases_with_higher_cg(self) -> None:
        """Reduce quasi-steady lateral capacity when CoG height increases."""
        vehicle = sample_vehicle_parameters()
        tires = default_axle_tire_parameters()
        physics = SingleTrackPhysics(
            max_drive_accel=8.0,
            max_brake_accel=16.0,
            peak_slip_angle=0.12,
        )
        numerics = SingleTrackNumerics()
        low_cg_model = build_single_track_model(
            vehicle=replace(vehicle, cg_height=vehicle.cg_height * 0.9),
            tires=tires,
            physics=physics,
            numerics=numerics,
        )
        high_cg_model = build_single_track_model(
            vehicle=replace(vehicle, cg_height=vehicle.cg_height * 1.1),
            tires=tires,
            physics=physics,
            numerics=numerics,
        )

        low_cg_limit = low_cg_model.lateral_accel_limit(speed=45.0, banking=0.0)
        high_cg_limit = high_cg_model.lateral_accel_limit(speed=45.0, banking=0.0)
        self.assertLess(high_cg_limit, low_cg_limit)

    def test_single_track_lateral_capacity_increases_with_wider_track(self) -> None:
        """Increase quasi-steady lateral capacity when track widths increase."""
        vehicle = sample_vehicle_parameters()
        tires = default_axle_tire_parameters()
        physics = SingleTrackPhysics(
            max_drive_accel=8.0,
            max_brake_accel=16.0,
            peak_slip_angle=0.12,
        )
        numerics = SingleTrackNumerics()
        narrow_model = build_single_track_model(
            vehicle=replace(
                vehicle,
                front_track=vehicle.front_track * 0.9,
                rear_track=vehicle.rear_track * 0.9,
            ),
            tires=tires,
            physics=physics,
            numerics=numerics,
        )
        wide_model = build_single_track_model(
            vehicle=replace(
                vehicle,
                front_track=vehicle.front_track * 1.1,
                rear_track=vehicle.rear_track * 1.1,
            ),
            tires=tires,
            physics=physics,
            numerics=numerics,
        )

        narrow_limit = narrow_model.lateral_accel_limit(speed=45.0, banking=0.0)
        wide_limit = wide_model.lateral_accel_limit(speed=45.0, banking=0.0)
        self.assertGreater(wide_limit, narrow_limit)

    def test_axle_tire_load_helpers_match_vertical_load_numpy(self) -> None:
        """Match axle-tire helper load sums to total static-plus-aero load (NumPy)."""
        vehicle = sample_vehicle_parameters()
        speed = np.array([0.0, 30.0, 60.0], dtype=float)
        downforce_scale = (
            0.5 * vehicle.air_density * vehicle.lift_coefficient * vehicle.frontal_area
        )
        front_tire, rear_tire = axle_tire_loads_numpy(
            speed=speed,
            mass=vehicle.mass,
            downforce_scale=downforce_scale,
            front_downforce_share=0.52,
            front_weight_fraction=vehicle.front_weight_fraction,
        )
        total_vertical = vehicle.mass * GRAVITY + downforce_scale * speed * speed
        np.testing.assert_allclose(
            2.0 * (front_tire + rear_tire),
            total_vertical,
            rtol=0.0,
            atol=1e-9,
        )
        self.assertTrue(np.all(front_tire > 0.0))
        self.assertTrue(np.all(rear_tire > 0.0))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_axle_tire_load_helpers_match_vertical_load_torch(self) -> None:
        """Match axle-tire helper load sums to total static-plus-aero load (Torch)."""
        import torch

        vehicle = sample_vehicle_parameters()
        speed = torch.tensor([0.0, 30.0, 60.0], dtype=torch.float64)
        downforce_scale = (
            0.5 * vehicle.air_density * vehicle.lift_coefficient * vehicle.frontal_area
        )
        front_tire, rear_tire = axle_tire_loads_torch(
            torch=torch,
            speed=speed,
            mass=vehicle.mass,
            downforce_scale=downforce_scale,
            front_downforce_share=0.52,
            front_weight_fraction=vehicle.front_weight_fraction,
        )
        total_vertical = vehicle.mass * GRAVITY + downforce_scale * speed * speed
        torch.testing.assert_close(
            2.0 * (front_tire + rear_tire),
            total_vertical,
            rtol=0.0,
            atol=1e-9,
        )
        self.assertTrue(bool(torch.all(front_tire > 0.0).item()))
        self.assertTrue(bool(torch.all(rear_tire > 0.0).item()))


if __name__ == "__main__":
    unittest.main()
