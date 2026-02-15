"""3-DOF bicycle dynamics model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lap_time_sim.tire.models import AxleTireParameters
from lap_time_sim.tire.pacejka import axle_lateral_forces
from lap_time_sim.utils.constants import SMALL_EPS
from lap_time_sim.vehicle.aero import aero_forces
from lap_time_sim.vehicle.load_transfer import estimate_normal_loads
from lap_time_sim.vehicle.params import VehicleParameters

MIN_SLIP_ANGLE_REFERENCE_SPEED_MPS = 0.5


@dataclass(frozen=True)
class VehicleState:
    """Vehicle state for the bicycle model."""

    vx_mps: float
    vy_mps: float
    yaw_rate_rps: float


@dataclass(frozen=True)
class ControlInput:
    """Control inputs for the bicycle model."""

    steer_rad: float
    longitudinal_accel_cmd_mps2: float


@dataclass(frozen=True)
class ForceBalance:
    """Force-balance quantities used for analysis and integration."""

    alpha_front_rad: float
    alpha_rear_rad: float
    fy_front_n: float
    fy_rear_n: float
    yaw_moment_nm: float


class BicycleModel:
    """3-DOF bicycle model using lateral Pacejka tire forces."""

    def __init__(self, vehicle: VehicleParameters, tires: AxleTireParameters) -> None:
        self.vehicle = vehicle
        self.tires = tires
        self.vehicle.validate()
        self.tires.validate()

    def slip_angles(self, state: VehicleState, steer_rad: float) -> tuple[float, float]:
        """Compute front and rear slip angles."""
        u = max(abs(state.vx_mps), MIN_SLIP_ANGLE_REFERENCE_SPEED_MPS)
        a = self.vehicle.cg_to_front_axle_m
        b = self.vehicle.cg_to_rear_axle_m

        alpha_front = steer_rad - np.arctan2(state.vy_mps + a * state.yaw_rate_rps, u)
        alpha_rear = -np.arctan2(state.vy_mps - b * state.yaw_rate_rps, u)
        return float(alpha_front), float(alpha_rear)

    def force_balance(self, state: VehicleState, control: ControlInput) -> ForceBalance:
        """Evaluate lateral forces and yaw moment for the current state."""
        alpha_front, alpha_rear = self.slip_angles(state, control.steer_rad)
        loads = estimate_normal_loads(
            self.vehicle,
            speed_mps=state.vx_mps,
            longitudinal_accel_mps2=control.longitudinal_accel_cmd_mps2,
            lateral_accel_mps2=0.0,
        )

        fy_front, fy_rear = axle_lateral_forces(
            front_slip_rad=alpha_front,
            rear_slip_rad=alpha_rear,
            front_axle_load_n=loads.front_axle_n,
            rear_axle_load_n=loads.rear_axle_n,
            axle_params=self.tires,
        )

        yaw_moment = (
            self.vehicle.cg_to_front_axle_m * fy_front * np.cos(control.steer_rad)
            - self.vehicle.cg_to_rear_axle_m * fy_rear
        )
        return ForceBalance(
            alpha_front_rad=alpha_front,
            alpha_rear_rad=alpha_rear,
            fy_front_n=fy_front,
            fy_rear_n=fy_rear,
            yaw_moment_nm=float(yaw_moment),
        )

    def derivatives(self, state: VehicleState, control: ControlInput) -> VehicleState:
        """Compute time derivatives for state integration."""
        fb = self.force_balance(state, control)
        aero = aero_forces(self.vehicle, state.vx_mps)

        mass = self.vehicle.mass_kg
        fx_n = mass * control.longitudinal_accel_cmd_mps2 - aero.drag_n

        vx = state.vx_mps
        vy = state.vy_mps
        yaw = state.yaw_rate_rps

        dvx = (fx_n - fb.fy_front_n * np.sin(control.steer_rad) + mass * vy * yaw) / mass
        dvy = (fb.fy_rear_n + fb.fy_front_n * np.cos(control.steer_rad) - mass * vx * yaw) / mass
        dyaw = fb.yaw_moment_nm / self.vehicle.yaw_inertia_kgm2

        return VehicleState(vx_mps=float(dvx), vy_mps=float(dvy), yaw_rate_rps=float(dyaw))

    @staticmethod
    def to_array(state: VehicleState) -> np.ndarray:
        """Convert state dataclass to ndarray."""
        return np.array([state.vx_mps, state.vy_mps, state.yaw_rate_rps], dtype=float)

    @staticmethod
    def from_array(values: np.ndarray) -> VehicleState:
        """Convert ndarray to state dataclass."""
        return VehicleState(
            vx_mps=float(values[0]),
            vy_mps=float(values[1]),
            yaw_rate_rps=float(values[2]),
        )

    @staticmethod
    def sanitize_speed(speed_mps: float) -> float:
        """Avoid zero speed in slip-angle calculations."""
        return max(speed_mps, SMALL_EPS)
