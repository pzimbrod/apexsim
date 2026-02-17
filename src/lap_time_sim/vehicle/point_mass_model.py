"""Solver-facing point-mass vehicle model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lap_time_sim.simulation.model_api import VehicleModelDiagnostics
from lap_time_sim.utils.constants import GRAVITY, SMALL_EPS
from lap_time_sim.utils.exceptions import ConfigurationError
from lap_time_sim.vehicle._model_base import EnvelopeVehicleModel
from lap_time_sim.vehicle._physics_primitives import EnvelopePhysics
from lap_time_sim.vehicle.aero import aero_forces
from lap_time_sim.vehicle.params import VehicleParameters

if TYPE_CHECKING:
    from lap_time_sim.tire.models import AxleTireParameters
    from lap_time_sim.vehicle.bicycle_model import BicycleNumerics, BicyclePhysics

DEFAULT_CALIBRATION_SAMPLE_COUNT = 50


@dataclass(frozen=True)
class PointMassPhysics:
    """Physical inputs for the point-mass lap-time model.

    Args:
        max_drive_accel: Maximum forward tire acceleration on flat road and zero
            lateral demand, excluding drag and grade [m/s^2].
        max_brake_accel: Maximum braking deceleration magnitude on flat road and
            zero lateral demand, excluding drag and grade [m/s^2].
        friction_coefficient: Isotropic tire-road friction coefficient used for
            lateral limit and friction-circle coupling (-).
    """

    max_drive_accel: float = 8.0
    max_brake_accel: float = 16.0
    friction_coefficient: float = 1.70

    @property
    def envelope(self) -> EnvelopePhysics:
        """Return shared longitudinal envelope limits.

        Returns:
            Internal shared envelope-limit representation used by multiple
            solver model families.
        """
        return EnvelopePhysics(
            max_drive_accel=self.max_drive_accel,
            max_brake_accel=self.max_brake_accel,
        )

    def validate(self) -> None:
        """Validate physical model parameters.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If limits or
                friction settings violate required bounds.
        """
        self.envelope.validate()
        if self.friction_coefficient <= 0.0:
            msg = "friction_coefficient must be positive"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class PointMassCalibrationResult:
    """Calibration outputs for matching point-mass lateral limits to bicycle behavior.

    Args:
        friction_coefficient: Identified isotropic friction coefficient (-).
        speed_samples: Calibration speed samples used by the identification [m/s].
        bicycle_lateral_limit: Bicycle-model lateral limit evaluated at
            ``speed_samples`` [m/s^2].
        normal_accel_limit: Point-mass normal acceleration budget evaluated at
            ``speed_samples`` [m/s^2].
        mu_samples: Per-sample effective friction ratios computed as
            ``bicycle_lateral_limit / normal_accel_limit`` (-).
    """

    friction_coefficient: float
    speed_samples: np.ndarray
    bicycle_lateral_limit: np.ndarray
    normal_accel_limit: np.ndarray
    mu_samples: np.ndarray


class PointMassModel(EnvelopeVehicleModel):
    """Vehicle-model API implementation for a point-mass backend."""

    def __init__(self, vehicle: VehicleParameters, physics: PointMassPhysics) -> None:
        """Initialize point-mass solver backend.

        Args:
            vehicle: Vehicle parameterization for mass, aero, and axle split.
            physics: Point-mass physical model settings.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If any provided
                parameter set is invalid.
        """
        self.vehicle = vehicle
        self.physics = physics
        super().__init__(vehicle=vehicle, envelope_physics=physics.envelope)
        self.validate()

    def _validate_backend(self) -> None:
        """Validate point-mass-specific model configuration.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If point-mass
                physical parameters violate constraints.
        """
        self.physics.validate()

    def _normal_accel_limit(self, speed: float) -> float:
        """Compute available tire normal acceleration from weight and downforce.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Positive normal-acceleration equivalent available at tire contact
            patch as ``(weight + downforce) / mass`` [m/s^2].
        """
        downforce = aero_forces(self.vehicle, speed).downforce
        return max(GRAVITY + downforce / self.vehicle.mass, SMALL_EPS)

    def _tire_accel_limit(self, speed: float) -> float:
        """Compute isotropic tire acceleration limit at a speed operating point.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Tire acceleration magnitude limit from isotropic friction [m/s^2].
        """
        return self.physics.friction_coefficient * self._normal_accel_limit(speed)

    def lateral_accel_limit(self, speed: float, banking: float) -> float:
        """Estimate lateral acceleration capacity for the operating point.

        Args:
            speed: Vehicle speed [m/s].
            banking: Track banking angle [rad].

        Returns:
            Quasi-steady lateral acceleration limit [m/s^2].
        """
        ay_tire = self._tire_accel_limit(speed)
        ay_banking = GRAVITY * float(np.sin(banking))
        return float(max(ay_tire + ay_banking, SMALL_EPS))

    def max_longitudinal_accel(
        self,
        speed: float,
        lateral_accel_required: float,
        grade: float,
        banking: float,
    ) -> float:
        """Compute net forward acceleration limit along path tangent.

        Args:
            speed: Vehicle speed [m/s].
            lateral_accel_required: Required lateral acceleration magnitude [m/s^2].
            grade: Track grade defined as ``dz/ds``.
            banking: Track banking angle [rad].

        Returns:
            Net forward acceleration along path tangent [m/s^2].
        """
        ay_limit = self.lateral_accel_limit(speed, banking)
        circle_scale = self._friction_circle_scale(lateral_accel_required, ay_limit)
        tire_limit = min(self._tire_accel_limit(speed), self.envelope_physics.max_drive_accel)
        tire_accel = tire_limit * circle_scale

        return self._net_forward_accel(tire_accel=tire_accel, speed=speed, grade=grade)

    def max_longitudinal_decel(
        self,
        speed: float,
        lateral_accel_required: float,
        grade: float,
        banking: float,
    ) -> float:
        """Compute available deceleration magnitude along path tangent.

        Args:
            speed: Vehicle speed [m/s].
            lateral_accel_required: Required lateral acceleration magnitude [m/s^2].
            grade: Track grade defined as ``dz/ds``.
            banking: Track banking angle [rad].

        Returns:
            Non-negative deceleration magnitude along path tangent [m/s^2].
        """
        ay_limit = self.lateral_accel_limit(speed, banking)
        circle_scale = self._friction_circle_scale(lateral_accel_required, ay_limit)
        tire_limit = min(self._tire_accel_limit(speed), self.envelope_physics.max_brake_accel)
        tire_brake = tire_limit * circle_scale

        return self._net_brake_decel(tire_brake=tire_brake, speed=speed, grade=grade)

    def diagnostics(
        self,
        speed: float,
        longitudinal_accel: float,
        lateral_accel: float,
        curvature: float,
    ) -> VehicleModelDiagnostics:
        """Evaluate point-mass diagnostics for analysis outputs.

        Args:
            speed: Vehicle speed [m/s].
            longitudinal_accel: Net longitudinal acceleration [m/s^2].
            lateral_accel: Lateral acceleration [m/s^2].
            curvature: Path curvature [1/m].

        Returns:
            Diagnostic values for plotting and KPI post-processing.
        """
        del lateral_accel, curvature
        aero = aero_forces(self.vehicle, speed)
        weight = self.vehicle.mass * GRAVITY

        front_axle = weight * self.vehicle.front_weight_fraction + aero.front_downforce
        rear_axle = weight * (1.0 - self.vehicle.front_weight_fraction) + aero.rear_downforce

        tractive_force = self.vehicle.mass * longitudinal_accel + aero.drag
        power = tractive_force * speed

        return VehicleModelDiagnostics(
            yaw_moment=0.0,
            front_axle_load=float(front_axle),
            rear_axle_load=float(rear_axle),
            power=float(power),
        )


def build_point_mass_model(
    vehicle: VehicleParameters,
    physics: PointMassPhysics | None = None,
) -> PointMassModel:
    """Build a point-mass solver model with sensible defaults.

    Args:
        vehicle: Vehicle parameterization for mass, aero, and axle split.
        physics: Optional physical model settings. Defaults to
            :class:`PointMassPhysics`.

    Returns:
        Fully validated solver-facing point-mass model.
    """
    return PointMassModel(vehicle=vehicle, physics=physics or PointMassPhysics())


def calibrate_point_mass_friction_to_bicycle(
    vehicle: VehicleParameters,
    tires: AxleTireParameters,
    bicycle_physics: BicyclePhysics | None = None,
    bicycle_numerics: BicycleNumerics | None = None,
    speed_samples: np.ndarray | None = None,
) -> PointMassCalibrationResult:
    """Calibrate point-mass friction coefficient to bicycle lateral capability.

    The calibration matches the point-mass isotropic lateral limit
    ``mu * (g + F_down(v)/m)`` to the bicycle model's quasi-steady lateral limit
    in a least-squares sense over provided speed samples.

    Args:
        vehicle: Vehicle parameterization shared by both models.
        tires: Tire parameters used by the bicycle model.
        bicycle_physics: Optional bicycle-physics settings for calibration.
            Defaults to :class:`lap_time_sim.vehicle.BicyclePhysics`.
        bicycle_numerics: Optional bicycle numerical settings for calibration.
            Defaults to :class:`lap_time_sim.vehicle.BicycleNumerics`.
        speed_samples: Optional calibration speeds [m/s]. If omitted, a linear
            sweep from 10 to 90 m/s is used.

    Returns:
        Calibration result with identified friction coefficient and
        intermediate traces.

    Raises:
        lap_time_sim.utils.exceptions.ConfigurationError: If provided speed
            samples are empty or contain non-positive entries.
    """
    from lap_time_sim.vehicle.bicycle_model import BicycleModel, BicycleNumerics, BicyclePhysics

    bicycle_model = BicycleModel(
        vehicle=vehicle,
        tires=tires,
        physics=bicycle_physics or BicyclePhysics(),
        numerics=bicycle_numerics or BicycleNumerics(),
    )
    bicycle_model.validate()

    if speed_samples is None:
        speeds = np.linspace(10.0, 90.0, DEFAULT_CALIBRATION_SAMPLE_COUNT, dtype=float)
    else:
        speeds = np.asarray(speed_samples, dtype=float)
        if speeds.size == 0:
            msg = "speed_samples must not be empty"
            raise ConfigurationError(msg)
        if np.any(speeds <= 0.0):
            msg = "speed_samples must be strictly positive"
            raise ConfigurationError(msg)

    bicycle_ay = np.array(
        [bicycle_model.lateral_accel_limit(float(v), 0.0) for v in speeds],
        dtype=float,
    )
    normal_accel = np.array(
        [GRAVITY + aero_forces(vehicle, float(v)).downforce / vehicle.mass for v in speeds],
        dtype=float,
    )
    normal_accel = np.maximum(normal_accel, SMALL_EPS)

    numerator = float(np.dot(normal_accel, bicycle_ay))
    denominator = float(np.dot(normal_accel, normal_accel))
    mu_fit = max(numerator / max(denominator, SMALL_EPS), SMALL_EPS)
    mu_samples = bicycle_ay / normal_accel

    return PointMassCalibrationResult(
        friction_coefficient=float(mu_fit),
        speed_samples=speeds,
        bicycle_lateral_limit=bicycle_ay,
        normal_accel_limit=normal_accel,
        mu_samples=mu_samples,
    )
