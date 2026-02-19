"""Physical equations for single-track models built on point-mass foundations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from apexsim.simulation.model_api import ModelDiagnostics
from apexsim.tire.models import AxleTireParameters
from apexsim.tire.pacejka import magic_formula_lateral
from apexsim.utils.constants import GRAVITY
from apexsim.vehicle._backend_physics_core import (
    roll_stiffness_front_share_numpy,
    single_track_wheel_loads_numpy,
)
from apexsim.vehicle._point_mass_physics import PointMassPhysicalMixin, PointMassPhysicalState
from apexsim.vehicle.single_track.dynamics import (
    SingleTrackDynamicsModel,
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

    def _iterate_lateral_limit(
        self,
        speed: np.ndarray,
        ay_banking: np.ndarray,
    ) -> np.ndarray:
        """Run fixed-point lateral-limit iteration with wheel-load-aware tire forces.

        Args:
            speed: Vehicle speed samples [m/s].
            ay_banking: Banking-induced lateral acceleration samples [m/s^2].

        Returns:
            Converged lateral-acceleration limit samples [m/s^2].
        """
        speed_array, ay_banking_array = np.broadcast_arrays(
            np.asarray(speed, dtype=float),
            np.asarray(ay_banking, dtype=float),
        )
        zero_longitudinal = np.zeros_like(speed_array, dtype=float)
        front_roll_share = roll_stiffness_front_share_numpy(
            front_spring_rate=self.vehicle.front_spring_rate,
            rear_spring_rate=self.vehicle.rear_spring_rate,
            front_arb_distribution=self.vehicle.front_arb_distribution,
        )
        ay_estimate = np.full(
            np.shape(ay_banking_array),
            self.numerics.min_lateral_accel_limit,
            dtype=float,
        )
        for _ in range(self.numerics.lateral_limit_max_iterations):
            (
                _front_axle_load,
                _rear_axle_load,
                front_left_load,
                front_right_load,
                rear_left_load,
                rear_right_load,
            ) = single_track_wheel_loads_numpy(
                speed=speed_array,
                mass=self.vehicle.mass,
                downforce_scale=self._downforce_scale,
                front_downforce_share=self._front_downforce_share,
                front_weight_fraction=self.vehicle.front_weight_fraction,
                longitudinal_accel=zero_longitudinal,
                lateral_accel=ay_estimate,
                cg_height=self.vehicle.cg_height,
                wheelbase=self.vehicle.wheelbase,
                front_track=self.vehicle.front_track,
                rear_track=self.vehicle.rear_track,
                front_roll_stiffness_share=front_roll_share,
            )
            fy_front = np.asarray(
                magic_formula_lateral(
                    self._single_track_lateral_physics.peak_slip_angle,
                    front_left_load,
                    self.tires.front,
                ),
                dtype=float,
            ) + np.asarray(
                magic_formula_lateral(
                    self._single_track_lateral_physics.peak_slip_angle,
                    front_right_load,
                    self.tires.front,
                ),
                dtype=float,
            )
            fy_rear = np.asarray(
                magic_formula_lateral(
                    self._single_track_lateral_physics.peak_slip_angle,
                    rear_left_load,
                    self.tires.rear,
                ),
                dtype=float,
            ) + np.asarray(
                magic_formula_lateral(
                    self._single_track_lateral_physics.peak_slip_angle,
                    rear_right_load,
                    self.tires.rear,
                ),
                dtype=float,
            )
            ay_tire = (fy_front + fy_rear) / self.vehicle.mass
            ay_next = np.maximum(
                self.numerics.min_lateral_accel_limit,
                ay_tire + ay_banking_array,
            )

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
        speed_array = np.asarray(speed, dtype=float)
        ay_banking = np.asarray(GRAVITY * float(np.sin(banking)), dtype=float)
        ay_limit = self._iterate_lateral_limit(
            speed=speed_array,
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
        return self._iterate_lateral_limit(
            speed=speed_array,
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
        return float(self._scaled_drive_envelope_accel_limit())

    def _brake_tire_accel_limit(self, speed: float) -> float:
        """Return pre-envelope tire-limited brake deceleration for single-track model.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Pre-envelope brake deceleration limit [m/s^2].
        """
        del speed
        return float(self._scaled_brake_envelope_accel_limit())

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
        loads = estimate_normal_loads(
            self.vehicle,
            speed=speed,
            longitudinal_accel=longitudinal_accel,
            lateral_accel=lateral_accel,
        )
        power = self._tractive_power(speed=speed, longitudinal_accel=longitudinal_accel)

        return ModelDiagnostics(
            yaw_moment=0.0,
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
        speed_array, accel_array, lateral_array, _curvature_array = np.broadcast_arrays(
            np.asarray(speed, dtype=float),
            np.asarray(longitudinal_accel, dtype=float),
            np.asarray(lateral_accel, dtype=float),
            np.asarray(curvature, dtype=float),
        )
        shape = speed_array.shape

        yaw_moment = np.zeros_like(speed_array, dtype=float)
        front_axle = np.empty_like(speed_array, dtype=float)
        rear_axle = np.empty_like(speed_array, dtype=float)

        speed_flat = speed_array.ravel()
        accel_flat = accel_array.ravel()
        lateral_flat = lateral_array.ravel()
        front_flat = front_axle.ravel()
        rear_flat = rear_axle.ravel()

        for idx in range(speed_flat.size):
            speed_value = float(speed_flat[idx])
            accel_value = float(accel_flat[idx])
            lateral_value = float(lateral_flat[idx])

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
