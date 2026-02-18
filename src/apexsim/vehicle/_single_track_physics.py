"""Physical equations for single-track models built on point-mass foundations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from apexsim.simulation.model_api import ModelDiagnostics
from apexsim.tire.models import AxleTireParameters
from apexsim.tire.pacejka import magic_formula_lateral
from apexsim.utils.constants import GRAVITY, SMALL_EPS
from apexsim.vehicle._point_mass_physics import PointMassPhysicalMixin, PointMassPhysicalState
from apexsim.vehicle.single_track.dynamics import (
    ControlInput,
    SingleTrackDynamicsModel,
    VehicleState,
)
from apexsim.vehicle.single_track.load_transfer import estimate_normal_loads


class SingleTrackLateralPhysicsProtocol(Protocol):
    """Protocol for single-track-specific lateral physics settings."""

    @property
    def peak_slip_angle(self) -> float:
        """Return quasi-steady peak slip angle.

        Returns:
            Quasi-steady peak slip angle [rad].
        """

    def validate(self) -> None:
        """Validate lateral-physics settings."""


class SingleTrackNumericsProtocol(Protocol):
    """Protocol for single-track lateral-iteration numerical controls."""

    @property
    def min_lateral_accel_limit(self) -> float:
        """Return minimum lateral-acceleration limit.

        Returns:
            Minimum lateral-acceleration limit [m/s^2].
        """

    @property
    def lateral_limit_max_iterations(self) -> int:
        """Return maximum fixed-point iteration count.

        Returns:
            Maximum fixed-point iteration count.
        """

    @property
    def lateral_limit_convergence_tolerance(self) -> float:
        """Return fixed-point convergence tolerance.

        Returns:
            Fixed-point convergence tolerance [m/s^2].
        """

    def validate(self) -> None:
        """Validate numerical settings."""


class SingleTrackPhysicalState(PointMassPhysicalState, Protocol):
    """Protocol describing state shared by single-track physical mixins."""

    tires: AxleTireParameters
    numerics: SingleTrackNumericsProtocol
    _single_track_lateral_physics: SingleTrackLateralPhysicsProtocol
    _dynamics: SingleTrackDynamicsModel


class SingleTrackPhysicalMixin(PointMassPhysicalMixin):
    """SingleTrack physical extension on top of point-mass physical foundations."""

    if TYPE_CHECKING:
        tires: AxleTireParameters
        numerics: SingleTrackNumericsProtocol
        _single_track_lateral_physics: SingleTrackLateralPhysicsProtocol
        _dynamics: SingleTrackDynamicsModel

    def _axle_tire_loads(
        self,
        speed: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate front/rear per-tire normal loads for envelope evaluation.

        Args:
            speed: Vehicle speed sample or vector [m/s].

        Returns:
            Tuple ``(front_tire_load, rear_tire_load)`` [N].
        """
        speed_array = np.asarray(speed, dtype=float)
        downforce_total = self._downforce_total_batch(speed_array)
        front_downforce = downforce_total * self._front_downforce_share

        weight = self.vehicle.mass * GRAVITY
        total_vertical_load = weight + downforce_total
        front_static_load = weight * self.vehicle.front_weight_fraction

        min_axle_load = 2.0 * SMALL_EPS
        front_axle_raw = front_static_load + front_downforce
        front_axle_load = np.clip(
            front_axle_raw,
            min_axle_load,
            total_vertical_load - min_axle_load,
        )
        rear_axle_load = total_vertical_load - front_axle_load

        front_tire_load = np.maximum(front_axle_load * 0.5, SMALL_EPS)
        rear_tire_load = np.maximum(rear_axle_load * 0.5, SMALL_EPS)
        return np.asarray(front_tire_load, dtype=float), np.asarray(rear_tire_load, dtype=float)

    def _iterate_lateral_limit(
        self,
        front_tire_load: np.ndarray,
        rear_tire_load: np.ndarray,
        ay_banking: np.ndarray,
    ) -> np.ndarray:
        """Run fixed-point lateral-limit iteration for vectorized tire loads.

        Args:
            front_tire_load: Front per-tire normal-load samples [N].
            rear_tire_load: Rear per-tire normal-load samples [N].
            ay_banking: Banking-induced lateral acceleration samples [m/s^2].

        Returns:
            Converged lateral-acceleration limit samples [m/s^2].
        """
        ay_estimate = np.full(
            np.shape(ay_banking),
            self.numerics.min_lateral_accel_limit,
            dtype=float,
        )
        for _ in range(self.numerics.lateral_limit_max_iterations):
            fy_front = 2.0 * np.asarray(
                magic_formula_lateral(
                    self._single_track_lateral_physics.peak_slip_angle,
                    front_tire_load,
                    self.tires.front,
                ),
                dtype=float,
            )
            fy_rear = 2.0 * np.asarray(
                magic_formula_lateral(
                    self._single_track_lateral_physics.peak_slip_angle,
                    rear_tire_load,
                    self.tires.rear,
                ),
                dtype=float,
            )
            ay_tire = (fy_front + fy_rear) / self.vehicle.mass
            ay_next = np.maximum(self.numerics.min_lateral_accel_limit, ay_tire + ay_banking)

            if (
                float(np.max(np.abs(ay_next - ay_estimate)))
                <= self.numerics.lateral_limit_convergence_tolerance
            ):
                ay_estimate = ay_next
                break
            ay_estimate = ay_next
        return np.asarray(ay_estimate, dtype=float)

    def lateral_accel_limit(
        self,
        speed: float,
        banking: float,
    ) -> float:
        """Estimate lateral acceleration capacity for scalar operating point.

        Args:
            speed: Vehicle speed [m/s].
            banking: Track banking angle [rad].

        Returns:
            Quasi-steady lateral acceleration limit [m/s^2].
        """
        front_tire_load, rear_tire_load = self._axle_tire_loads(speed)
        ay_banking = np.asarray(GRAVITY * float(np.sin(banking)), dtype=float)
        ay_limit = self._iterate_lateral_limit(
            front_tire_load=np.asarray(front_tire_load, dtype=float),
            rear_tire_load=np.asarray(rear_tire_load, dtype=float),
            ay_banking=ay_banking,
        )
        return float(ay_limit)

    def lateral_accel_limit_batch(
        self,
        speed: np.ndarray,
        banking: np.ndarray,
    ) -> np.ndarray:
        """Estimate lateral acceleration limits for vectorized operating points.

        Args:
            speed: Speed samples [m/s].
            banking: Banking-angle samples [rad].

        Returns:
            Quasi-steady lateral acceleration limits [m/s^2].
        """
        speed_array, banking_array = np.broadcast_arrays(
            np.asarray(speed, dtype=float),
            np.asarray(banking, dtype=float),
        )
        ay_banking = GRAVITY * np.sin(banking_array)
        front_tire_load, rear_tire_load = self._axle_tire_loads(speed_array)
        return self._iterate_lateral_limit(
            front_tire_load=front_tire_load,
            rear_tire_load=rear_tire_load,
            ay_banking=ay_banking,
        )

    def _drive_tire_accel_limit(self, speed: float) -> float:
        """Return pre-envelope tire-limited drive acceleration for single-track model.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Pre-envelope drive acceleration limit [m/s^2].
        """
        del speed
        return float(self.envelope_physics.max_drive_accel)

    def _brake_tire_accel_limit(self, speed: float) -> float:
        """Return pre-envelope tire-limited brake deceleration for single-track model.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Pre-envelope brake deceleration limit [m/s^2].
        """
        del speed
        return float(self.envelope_physics.max_brake_accel)

    def diagnostics(
        self,
        speed: float,
        longitudinal_accel: float,
        lateral_accel: float,
        curvature: float,
    ) -> ModelDiagnostics:
        """Evaluate single-track diagnostics for post-processing outputs.

        Args:
            speed: Vehicle speed [m/s].
            longitudinal_accel: Net longitudinal acceleration [m/s^2].
            lateral_accel: Lateral acceleration [m/s^2].
            curvature: Path curvature [1/m].

        Returns:
            Diagnostic values for plotting and KPI post-processing.
        """
        steer = float(np.arctan(self.vehicle.wheelbase * curvature))
        state = VehicleState(vx=speed, vy=0.0, yaw_rate=speed * curvature)
        control = ControlInput(steer=steer, longitudinal_accel_cmd=longitudinal_accel)
        force_balance = self._dynamics.force_balance(state, control)

        loads = estimate_normal_loads(
            self.vehicle,
            speed=speed,
            longitudinal_accel=longitudinal_accel,
            lateral_accel=lateral_accel,
        )
        power = self._tractive_power(speed=speed, longitudinal_accel=longitudinal_accel)

        return ModelDiagnostics(
            yaw_moment=force_balance.yaw_moment,
            front_axle_load=loads.front_axle_load,
            rear_axle_load=loads.rear_axle_load,
            power=power,
        )

    def diagnostics_batch(
        self,
        speed: np.ndarray,
        longitudinal_accel: np.ndarray,
        lateral_accel: np.ndarray,
        curvature: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate single-track diagnostics over vectorized operating points.

        Args:
            speed: Speed samples [m/s].
            longitudinal_accel: Net longitudinal-acceleration samples [m/s^2].
            lateral_accel: Lateral-acceleration samples [m/s^2].
            curvature: Curvature samples [1/m].

        Returns:
            Tuple ``(yaw_moment, front_axle_load, rear_axle_load, power)`` with
            arrays aligned to the input shape.
        """
        speed_array, accel_array, lateral_array, curvature_array = np.broadcast_arrays(
            np.asarray(speed, dtype=float),
            np.asarray(longitudinal_accel, dtype=float),
            np.asarray(lateral_accel, dtype=float),
            np.asarray(curvature, dtype=float),
        )
        shape = speed_array.shape

        yaw_moment = np.empty_like(speed_array, dtype=float)
        front_axle = np.empty_like(speed_array, dtype=float)
        rear_axle = np.empty_like(speed_array, dtype=float)

        speed_flat = speed_array.ravel()
        accel_flat = accel_array.ravel()
        lateral_flat = lateral_array.ravel()
        curvature_flat = curvature_array.ravel()
        yaw_flat = yaw_moment.ravel()
        front_flat = front_axle.ravel()
        rear_flat = rear_axle.ravel()

        for idx in range(speed_flat.size):
            speed_value = float(speed_flat[idx])
            accel_value = float(accel_flat[idx])
            lateral_value = float(lateral_flat[idx])
            curvature_value = float(curvature_flat[idx])

            steer = float(np.arctan(self.vehicle.wheelbase * curvature_value))
            state = VehicleState(vx=speed_value, vy=0.0, yaw_rate=speed_value * curvature_value)
            control = ControlInput(steer=steer, longitudinal_accel_cmd=accel_value)
            force_balance = self._dynamics.force_balance(state, control)
            yaw_flat[idx] = force_balance.yaw_moment

            loads = estimate_normal_loads(
                self.vehicle,
                speed=speed_value,
                longitudinal_accel=accel_value,
                lateral_accel=lateral_value,
            )
            front_flat[idx] = loads.front_axle_load
            rear_flat[idx] = loads.rear_axle_load

        power = self._tractive_power_batch(speed=speed_array, longitudinal_accel=accel_array)
        return (
            yaw_moment.reshape(shape),
            front_axle.reshape(shape),
            rear_axle.reshape(shape),
            np.asarray(power, dtype=float).reshape(shape),
        )
