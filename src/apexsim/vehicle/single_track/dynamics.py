"""3-DOF single-track dynamics model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apexsim.tire.models import AxleTireParameters
from apexsim.tire.pacejka import magic_formula_lateral
from apexsim.utils.constants import SMALL_EPS
from apexsim.vehicle.params import VehicleParameters
from apexsim.vehicle.single_track.load_transfer import estimate_normal_loads

MIN_SLIP_ANGLE_REFERENCE_SPEED = 0.5


@dataclass(frozen=True)
class VehicleState:
    """Vehicle state for the single-track model.

    Args:
        vx: Longitudinal velocity in body frame [m/s].
        vy: Lateral velocity in body frame [m/s].
        yaw_rate: Yaw rate [rad/s].
    """

    vx: float
    vy: float
    yaw_rate: float


@dataclass(frozen=True)
class ControlInput:
    """Control inputs for the single-track model.

    Args:
        steer: Front-wheel steering angle [rad].
        longitudinal_accel_cmd: Commanded longitudinal acceleration [m/s^2].
    """

    steer: float
    longitudinal_accel_cmd: float


@dataclass(frozen=True)
class ForceBalance:
    """Force-balance quantities used for analysis and integration.

    Args:
        front_slip_angle: Front equivalent slip angle [rad].
        rear_slip_angle: Rear equivalent slip angle [rad].
        front_lateral_force: Front-axle lateral tire force [N].
        rear_lateral_force: Rear-axle lateral tire force [N].
        yaw_moment: Net yaw moment about center of gravity [N*m].
    """

    front_slip_angle: float
    rear_slip_angle: float
    front_lateral_force: float
    rear_lateral_force: float
    yaw_moment: float


class SingleTrackDynamicsModel:
    """3-DOF single-track dynamics model using lateral Pacejka tire forces."""

    def __init__(self, vehicle: VehicleParameters, tires: AxleTireParameters) -> None:
        """Initialize the single-track model with validated parameters.

        Args:
            vehicle: Vehicle and chassis parameterization.
            tires: Front/rear Pacejka parameter sets.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If vehicle or tire
                parameters are invalid.
        """
        self.vehicle = vehicle
        self.tires = tires
        self.vehicle.validate()
        self.tires.validate()

    def slip_angles(self, state: VehicleState, steer: float) -> tuple[float, float]:
        """Compute front and rear axle slip angles.

        Args:
            state: Vehicle state ``(vx, vy, yaw_rate)``.
            steer: Front-wheel steering angle [rad].

        Returns:
            Tuple ``(front_slip_angle, rear_slip_angle)``.
        """
        u = max(abs(state.vx), MIN_SLIP_ANGLE_REFERENCE_SPEED)
        a = self.vehicle.cg_to_front_axle
        b = self.vehicle.cg_to_rear_axle

        alpha_front = steer - np.arctan2(state.vy + a * state.yaw_rate, u)
        alpha_rear = -np.arctan2(state.vy - b * state.yaw_rate, u)
        return float(alpha_front), float(alpha_rear)

    def force_balance(self, state: VehicleState, control: ControlInput) -> ForceBalance:
        """Evaluate lateral forces and yaw moment for the current state.

        Args:
            state: Vehicle state ``(vx, vy, yaw_rate)``.
            control: Steering and longitudinal acceleration command.

        Returns:
            Force-balance terms including slip angles, tire forces, and yaw moment.
        """
        alpha_front, alpha_rear = self.slip_angles(state, control.steer)
        body_speed = float(np.hypot(state.vx, state.vy))
        # Use a kinematic lateral-acceleration estimate for transient load transfer.
        lateral_accel_estimate = state.vx * state.yaw_rate
        loads = estimate_normal_loads(
            self.vehicle,
            speed=body_speed,
            longitudinal_accel=control.longitudinal_accel_cmd,
            lateral_accel=lateral_accel_estimate,
        )

        fy_front = float(
            magic_formula_lateral(alpha_front, loads.front_left_load, self.tires.front)
        ) + float(magic_formula_lateral(alpha_front, loads.front_right_load, self.tires.front))
        fy_rear = float(
            magic_formula_lateral(alpha_rear, loads.rear_left_load, self.tires.rear)
        ) + float(magic_formula_lateral(alpha_rear, loads.rear_right_load, self.tires.rear))

        yaw_moment = (
            self.vehicle.cg_to_front_axle * fy_front * np.cos(control.steer)
            - self.vehicle.cg_to_rear_axle * fy_rear
        )
        return ForceBalance(
            front_slip_angle=alpha_front,
            rear_slip_angle=alpha_rear,
            front_lateral_force=fy_front,
            rear_lateral_force=fy_rear,
            yaw_moment=float(yaw_moment),
        )

    def derivatives(self, state: VehicleState, control: ControlInput) -> VehicleState:
        """Compute time derivatives for state integration.

        Args:
            state: Vehicle state ``(vx, vy, yaw_rate)``.
            control: Steering and longitudinal acceleration command.

        Returns:
            Time derivatives ``(dvx/dt, dvy/dt, dr/dt)`` in SI units.
        """
        fb = self.force_balance(state, control)
        mass = self.vehicle.mass
        # The transient solver provides a *net* longitudinal acceleration command
        # (after envelope/drag/grade limits). Do not subtract aero drag again here.
        longitudinal_force = mass * control.longitudinal_accel_cmd

        vx = state.vx
        vy = state.vy
        yaw = state.yaw_rate

        dvx = (
            longitudinal_force
            - fb.front_lateral_force * np.sin(control.steer)
            + mass * vy * yaw
        ) / mass
        dvy = (
            fb.rear_lateral_force
            + fb.front_lateral_force * np.cos(control.steer)
            - mass * vx * yaw
        ) / mass
        dyaw = fb.yaw_moment / self.vehicle.yaw_inertia

        return VehicleState(vx=float(dvx), vy=float(dvy), yaw_rate=float(dyaw))

    @staticmethod
    def to_array(state: VehicleState) -> np.ndarray:
        """Convert state dataclass to ndarray.

        Args:
            state: Vehicle state dataclass.

        Returns:
            State vector ``[vx, vy, yaw_rate]``.
        """
        return np.array([state.vx, state.vy, state.yaw_rate], dtype=float)

    @staticmethod
    def from_array(values: np.ndarray) -> VehicleState:
        """Convert ndarray to state dataclass.

        Args:
            values: State vector with at least three elements.

        Returns:
            Parsed vehicle state dataclass.
        """
        return VehicleState(
            vx=float(values[0]),
            vy=float(values[1]),
            yaw_rate=float(values[2]),
        )

    @staticmethod
    def sanitize_speed(speed: float) -> float:
        """Apply a numerical lower bound to speed.

        Args:
            speed: Raw speed [m/s].

        Returns:
            Speed clamped to a positive lower bound for numerical stability.
        """
        return max(speed, SMALL_EPS)
