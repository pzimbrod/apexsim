"""Physical equations for point-mass-derived vehicle models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from apexsim.simulation.model_api import ModelDiagnostics
from apexsim.utils.constants import GRAVITY, SMALL_EPS
from apexsim.vehicle._physics_primitives import EnvelopePhysics
from apexsim.vehicle.params import VehicleParameters


class PointMassPhysicsProtocol(Protocol):
    """Protocol for point-mass friction inputs."""

    @property
    def friction_coefficient(self) -> float:
        """Return isotropic tire-road friction coefficient.

        Returns:
            Isotropic tire-road friction coefficient (-).
        """

    def validate(self) -> None:
        """Validate physical settings."""


class PointMassPhysicalState(Protocol):
    """Protocol describing state shared by point-mass-derived physical mixins."""

    vehicle: VehicleParameters
    envelope_physics: EnvelopePhysics
    _drag_force_scale: float
    _downforce_scale: float
    _front_downforce_share: float


class PointMassFrictionState(PointMassPhysicalState, Protocol):
    """Protocol for point-mass states with isotropic friction settings."""

    physics: PointMassPhysicsProtocol


class PointMassPhysicalMixin:
    """Backend-agnostic point-mass physical equations.

    This mixin is the reusable physical base for model families that follow the
    same envelope-style longitudinal coupling. More detailed models can inherit
    this class and override only the lateral-limit and tire-limit hooks.
    """

    if TYPE_CHECKING:
        vehicle: VehicleParameters
        envelope_physics: EnvelopePhysics
        physics: object
        _drag_force_scale: float
        _downforce_scale: float
        _front_downforce_share: float

        @staticmethod
        def _friction_circle_scale(
            lateral_accel_required: float,
            lateral_accel_limit: float,
        ) -> float: ...

        def _downforce_total(self, speed: float) -> float: ...

        def _downforce_front(self, speed: float) -> float: ...

        def _downforce_rear(self, speed: float) -> float: ...

        def _drag_force(self, speed: float) -> float: ...

        def _net_forward_accel(self, tire_accel: float, speed: float, grade: float) -> float: ...

        def _net_brake_decel(self, tire_brake: float, speed: float, grade: float) -> float: ...

    def _downforce_total_batch(self, speed: np.ndarray) -> np.ndarray:
        """Compute total aerodynamic downforce for vectorized speed samples.

        Args:
            speed: Speed samples [m/s].

        Returns:
            Total downforce samples [N].
        """
        speed_non_negative = np.maximum(np.asarray(speed, dtype=float), 0.0)
        speed_squared = speed_non_negative * speed_non_negative
        downforce = self._downforce_scale * speed_squared
        return np.asarray(downforce, dtype=float)

    def _normal_accel_limit(self, speed: float) -> float:
        """Compute normal-acceleration budget from gravity and downforce.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Available normal-acceleration budget [m/s^2].
        """
        downforce = self._downforce_total(speed)
        return float(max(GRAVITY + downforce / self.vehicle.mass, SMALL_EPS))

    def _normal_accel_limit_batch(self, speed: np.ndarray) -> np.ndarray:
        """Compute vectorized normal-acceleration budget samples.

        Args:
            speed: Speed samples [m/s].

        Returns:
            Available normal-acceleration budget samples [m/s^2].
        """
        downforce = self._downforce_total_batch(speed)
        normal_accel = GRAVITY + downforce / self.vehicle.mass
        return np.maximum(normal_accel, SMALL_EPS)

    def _tire_accel_limit(self, speed: float) -> float:
        """Compute isotropic tire acceleration limit for scalar speed.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Isotropic tire acceleration magnitude limit [m/s^2].
        """
        physics = cast(PointMassPhysicsProtocol, self.physics)
        return float(physics.friction_coefficient * self._normal_accel_limit(speed))

    def _tire_accel_limit_batch(self, speed: np.ndarray) -> np.ndarray:
        """Compute isotropic tire acceleration limits for speed vectors.

        Args:
            speed: Speed samples [m/s].

        Returns:
            Isotropic tire acceleration magnitude limits [m/s^2].
        """
        physics = cast(PointMassPhysicsProtocol, self.physics)
        return physics.friction_coefficient * self._normal_accel_limit_batch(speed)

    def _drive_tire_accel_limit(self, speed: float) -> float:
        """Return pre-envelope tire-limited drive acceleration at a speed.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Pre-envelope drive acceleration limit [m/s^2].
        """
        return float(self._tire_accel_limit(speed))

    def _brake_tire_accel_limit(self, speed: float) -> float:
        """Return pre-envelope tire-limited brake deceleration at a speed.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Pre-envelope brake deceleration limit [m/s^2].
        """
        return float(self._tire_accel_limit(speed))

    def _scaled_drive_envelope_accel_limit(self) -> float:
        """Return drive acceleration envelope scaled by optional reference mass.

        Returns:
            Drive acceleration envelope limit [m/s^2].
        """
        reference_mass = self.envelope_physics.reference_mass
        if reference_mass is None:
            return float(self.envelope_physics.max_drive_accel)
        return float(self.envelope_physics.max_drive_accel * reference_mass / self.vehicle.mass)

    def _scaled_brake_envelope_accel_limit(self) -> float:
        """Return brake deceleration envelope scaled by optional reference mass.

        Returns:
            Brake deceleration envelope limit [m/s^2].
        """
        reference_mass = self.envelope_physics.reference_mass
        if reference_mass is None:
            return float(self.envelope_physics.max_brake_accel)
        return float(self.envelope_physics.max_brake_accel * reference_mass / self.vehicle.mass)

    def _lateral_limit_for_longitudinal(
        self,
        speed: float,
        banking: float,
    ) -> float:
        """Return lateral limit used by longitudinal friction-circle coupling.

        Args:
            speed: Vehicle speed [m/s].
            banking: Track banking angle [rad].

        Returns:
            Lateral acceleration limit used in longitudinal coupling [m/s^2].
        """
        return float(self.lateral_accel_limit(speed=speed, banking=banking))

    def _tractive_power(
        self,
        speed: float,
        longitudinal_accel: float,
    ) -> float:
        """Compute tractive power from speed and longitudinal acceleration.

        Args:
            speed: Vehicle speed [m/s].
            longitudinal_accel: Net longitudinal acceleration [m/s^2].

        Returns:
            Tractive power [W].
        """
        tractive_force = self.vehicle.mass * longitudinal_accel + self._drag_force(speed)
        return float(tractive_force * speed)

    def _tractive_power_batch(
        self,
        speed: np.ndarray,
        longitudinal_accel: np.ndarray,
    ) -> np.ndarray:
        """Compute vectorized tractive power from speed and acceleration arrays.

        Args:
            speed: Vehicle speed samples [m/s].
            longitudinal_accel: Net longitudinal-acceleration samples [m/s^2].

        Returns:
            Tractive power samples [W].
        """
        speed_array = np.asarray(speed, dtype=float)
        accel_array = np.asarray(longitudinal_accel, dtype=float)
        speed_non_negative = np.maximum(speed_array, 0.0)
        speed_squared = speed_non_negative * speed_non_negative
        drag_force = self._drag_force_scale * speed_squared
        tractive_force = self.vehicle.mass * accel_array + drag_force
        return np.asarray(tractive_force * speed_array, dtype=float)

    def lateral_accel_limit(
        self,
        speed: float,
        banking: float,
    ) -> float:
        """Estimate lateral acceleration capacity for a scalar operating point.

        Args:
            speed: Vehicle speed [m/s].
            banking: Track banking angle [rad].

        Returns:
            Quasi-steady lateral acceleration limit [m/s^2].
        """
        ay_tire = self._tire_accel_limit(speed)
        ay_banking = GRAVITY * float(np.sin(banking))
        return float(max(ay_tire + ay_banking, SMALL_EPS))

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
        ay_tire = self._tire_accel_limit_batch(speed_array)
        ay_banking = GRAVITY * np.sin(banking_array)
        ay_limit = np.maximum(ay_tire + ay_banking, SMALL_EPS)
        return np.asarray(ay_limit, dtype=float)

    def max_longitudinal_accel(
        self,
        speed: float,
        lateral_accel_required: float,
        grade: float,
        banking: float,
    ) -> float:
        """Compute available net forward acceleration along the path tangent.

        Args:
            speed: Vehicle speed [m/s].
            lateral_accel_required: Required lateral acceleration magnitude [m/s^2].
            grade: Track grade defined as ``dz/ds``.
            banking: Track banking angle [rad].

        Returns:
            Net forward acceleration along path tangent [m/s^2].
        """
        ay_limit = self._lateral_limit_for_longitudinal(speed=speed, banking=banking)
        circle_scale = self._friction_circle_scale(lateral_accel_required, ay_limit)

        tire_limit = min(
            self._drive_tire_accel_limit(speed),
            self._scaled_drive_envelope_accel_limit(),
        )
        tire_accel = tire_limit * circle_scale
        return self._net_forward_accel(tire_accel=tire_accel, speed=speed, grade=grade)

    def max_longitudinal_decel(
        self,
        speed: float,
        lateral_accel_required: float,
        grade: float,
        banking: float,
    ) -> float:
        """Compute available braking deceleration magnitude along the path tangent.

        Args:
            speed: Vehicle speed [m/s].
            lateral_accel_required: Required lateral acceleration magnitude [m/s^2].
            grade: Track grade defined as ``dz/ds``.
            banking: Track banking angle [rad].

        Returns:
            Non-negative deceleration magnitude along path tangent [m/s^2].
        """
        ay_limit = self._lateral_limit_for_longitudinal(speed=speed, banking=banking)
        circle_scale = self._friction_circle_scale(lateral_accel_required, ay_limit)

        tire_limit = min(
            self._brake_tire_accel_limit(speed),
            self._scaled_brake_envelope_accel_limit(),
        )
        tire_brake = tire_limit * circle_scale
        return self._net_brake_decel(tire_brake=tire_brake, speed=speed, grade=grade)

    def diagnostics(
        self,
        speed: float,
        longitudinal_accel: float,
        lateral_accel: float,
        curvature: float,
    ) -> ModelDiagnostics:
        """Evaluate point-mass diagnostics for post-processing outputs.

        Args:
            speed: Vehicle speed [m/s].
            longitudinal_accel: Net longitudinal acceleration [m/s^2].
            lateral_accel: Lateral acceleration [m/s^2].
            curvature: Path curvature [1/m].

        Returns:
            Diagnostic values for plotting and KPI post-processing.
        """
        del lateral_accel, curvature

        weight = self.vehicle.mass * GRAVITY
        front_downforce = self._downforce_front(speed)
        rear_downforce = self._downforce_rear(speed)

        front_axle = weight * self.vehicle.front_weight_fraction + front_downforce
        rear_axle = weight * (1.0 - self.vehicle.front_weight_fraction) + rear_downforce
        power = self._tractive_power(speed=speed, longitudinal_accel=longitudinal_accel)

        return ModelDiagnostics(
            yaw_moment=0.0,
            front_axle_load=float(front_axle),
            rear_axle_load=float(rear_axle),
            power=float(power),
        )

    def diagnostics_batch(
        self,
        speed: np.ndarray,
        longitudinal_accel: np.ndarray,
        lateral_accel: np.ndarray,
        curvature: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate diagnostics for vectorized operating-point samples.

        Args:
            speed: Speed samples [m/s].
            longitudinal_accel: Net longitudinal-acceleration samples [m/s^2].
            lateral_accel: Lateral-acceleration samples [m/s^2].
            curvature: Curvature samples [1/m].

        Returns:
            Tuple ``(yaw_moment, front_axle_load, rear_axle_load, power)`` with
            arrays aligned to the input shape.
        """
        del lateral_accel, curvature

        speed_array = np.asarray(speed, dtype=float)
        accel_array = np.asarray(longitudinal_accel, dtype=float)

        downforce_total = self._downforce_total_batch(speed_array)
        front_downforce = downforce_total * self._front_downforce_share
        rear_downforce = downforce_total * (1.0 - self._front_downforce_share)

        weight = self.vehicle.mass * GRAVITY
        front_axle = weight * self.vehicle.front_weight_fraction + front_downforce
        rear_axle = weight * (1.0 - self.vehicle.front_weight_fraction) + rear_downforce

        power = self._tractive_power_batch(speed=speed_array, longitudinal_accel=accel_array)
        yaw_moment = np.zeros_like(speed_array, dtype=float)
        return (
            np.asarray(yaw_moment, dtype=float),
            np.asarray(front_axle, dtype=float),
            np.asarray(rear_axle, dtype=float),
            np.asarray(power, dtype=float),
        )
