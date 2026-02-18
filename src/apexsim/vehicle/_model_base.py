"""Abstract OOP base utilities for solver-facing vehicle models."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from apexsim.simulation.model_api import VehicleModelBase
from apexsim.utils.constants import GRAVITY, SMALL_EPS
from apexsim.vehicle._physics_primitives import EnvelopePhysics
from apexsim.vehicle.params import VehicleParameters


class EnvelopeVehicleModel(VehicleModelBase):
    """Shared base class for vehicle models with envelope-limited longitudinal force.

    This base class centralizes common quasi-steady logic used by single_track and
    point-mass backends:
    - parameter validation layering,
    - friction-circle longitudinal scaling,
    - drag/grade corrections for net path-tangent acceleration.
    """

    def __init__(self, vehicle: VehicleParameters, envelope_physics: EnvelopePhysics) -> None:
        """Initialize shared model state.

        Args:
            vehicle: Vehicle parameterization used by the backend.
            envelope_physics: Shared forward/brake envelope acceleration limits.
        """
        self.vehicle = vehicle
        self.envelope_physics = envelope_physics

        self._drag_force_scale = (
            0.5 * vehicle.air_density * vehicle.drag_coefficient * vehicle.frontal_area
        )
        self._downforce_scale = (
            0.5 * vehicle.air_density * vehicle.lift_coefficient * vehicle.frontal_area
        )

        cop_from_rear = vehicle.cg_to_rear_axle + vehicle.cop_position
        self._front_downforce_share = min(max(cop_from_rear / vehicle.wheelbase, 0.0), 1.0)

    def validate(self) -> None:
        """Validate shared and backend-specific model parameters.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If any shared or
                backend-specific parameter violates required constraints.
        """
        self.vehicle.validate()
        self.envelope_physics.validate()
        self._validate_backend()

    @abstractmethod
    def _validate_backend(self) -> None:
        """Validate backend-specific model configuration."""

    @staticmethod
    def _friction_circle_scale(
        lateral_accel_required: float,
        lateral_accel_limit: float,
    ) -> float:
        """Compute remaining longitudinal utilization from friction-circle usage.

        Args:
            lateral_accel_required: Required lateral acceleration magnitude [m/s^2].
            lateral_accel_limit: Available lateral acceleration limit [m/s^2].

        Returns:
            Scalar in ``[0, 1]`` reducing longitudinal capability.
        """
        if lateral_accel_limit <= SMALL_EPS:
            return 0.0
        usage = min(abs(lateral_accel_required) / lateral_accel_limit, 1.0)
        return float(np.sqrt(max(0.0, 1.0 - usage * usage)))

    def _drag_force(self, speed: float) -> float:
        """Compute aerodynamic drag force from precomputed coefficients.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Aerodynamic drag force [N].
        """
        speed_sq = max(speed, 0.0) ** 2
        return self._drag_force_scale * speed_sq

    def _downforce_total(self, speed: float) -> float:
        """Compute total aerodynamic downforce from precomputed coefficients.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Total aerodynamic downforce [N].
        """
        speed_sq = max(speed, 0.0) ** 2
        return self._downforce_scale * speed_sq

    def _downforce_front(self, speed: float) -> float:
        """Compute front-axle aerodynamic downforce share.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Front axle aerodynamic downforce [N].
        """
        return self._downforce_total(speed) * self._front_downforce_share

    def _downforce_rear(self, speed: float) -> float:
        """Compute rear-axle aerodynamic downforce share.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Rear axle aerodynamic downforce [N].
        """
        return self._downforce_total(speed) * (1.0 - self._front_downforce_share)

    def _net_forward_accel(self, tire_accel: float, speed: float, grade: float) -> float:
        """Convert tire-limited forward acceleration to net path-tangent acceleration.

        Args:
            tire_accel: Available tire-limited forward acceleration [m/s^2].
            speed: Vehicle speed [m/s].
            grade: Track grade ``dz/ds`` [-].

        Returns:
            Net forward acceleration along path tangent [m/s^2].
        """
        drag_accel = self._drag_force(speed) / self.vehicle.mass
        grade_accel = GRAVITY * grade
        return float(tire_accel - drag_accel - grade_accel)

    def _net_brake_decel(self, tire_brake: float, speed: float, grade: float) -> float:
        """Convert tire-limited braking to net path-tangent deceleration magnitude.

        Args:
            tire_brake: Available tire-limited deceleration magnitude [m/s^2].
            speed: Vehicle speed [m/s].
            grade: Track grade ``dz/ds`` [-].

        Returns:
            Non-negative deceleration magnitude along path tangent [m/s^2].
        """
        drag_accel = self._drag_force(speed) / self.vehicle.mass
        grade_accel = GRAVITY * grade
        return float(max(tire_brake + drag_accel + grade_accel, 0.0))
