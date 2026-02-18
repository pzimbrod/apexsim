"""Solver-facing point-mass vehicle model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pylapsim.utils.constants import GRAVITY, SMALL_EPS
from pylapsim.utils.exceptions import ConfigurationError
from pylapsim.vehicle._model_base import EnvelopeVehicleModel
from pylapsim.vehicle._physics_primitives import EnvelopePhysics
from pylapsim.vehicle._point_mass_backends import (
    PointMassNumbaBackendMixin,
    PointMassTorchBackendMixin,
)
from pylapsim.vehicle._point_mass_physics import PointMassPhysicalMixin
from pylapsim.vehicle.params import VehicleParameters

if TYPE_CHECKING:
    from pylapsim.tire.models import AxleTireParameters
    from pylapsim.vehicle.single_track_model import SingleTrackNumerics, SingleTrackPhysics

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
            pylapsim.utils.exceptions.ConfigurationError: If limits or
                friction settings violate required bounds.
        """
        self.envelope.validate()
        if self.friction_coefficient <= 0.0:
            msg = "friction_coefficient must be positive"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class PointMassCalibrationResult:
    """Calibration outputs for matching point-mass lateral limits to single_track behavior.

    Args:
        friction_coefficient: Identified isotropic friction coefficient (-).
        speed_samples: Calibration speed samples used by the identification [m/s].
        single_track_lateral_limit: SingleTrack-model lateral limit evaluated at
            ``speed_samples`` [m/s^2].
        normal_accel_limit: Point-mass normal acceleration budget evaluated at
            ``speed_samples`` [m/s^2].
        mu_samples: Per-sample effective friction ratios computed as
            ``single_track_lateral_limit / normal_accel_limit`` (-).
    """

    friction_coefficient: float
    speed_samples: np.ndarray
    single_track_lateral_limit: np.ndarray
    normal_accel_limit: np.ndarray
    mu_samples: np.ndarray


class PointMassModel(
    PointMassTorchBackendMixin,
    PointMassNumbaBackendMixin,
    PointMassPhysicalMixin,
    EnvelopeVehicleModel,
):
    """Vehicle-model API implementation for a point-mass backend.

    The class is intentionally composed from separate OOP layers:

    - ``PointMassPhysicalMixin`` for backend-agnostic physics,
    - ``PointMassNumbaBackendMixin`` for numba-specific solver adapter API,
    - ``PointMassTorchBackendMixin`` for torch-specific solver adapter API.
    """

    physics: PointMassPhysics

    def __init__(self, vehicle: VehicleParameters, physics: PointMassPhysics) -> None:
        """Initialize point-mass solver backend.

        Args:
            vehicle: Vehicle parameterization for mass, aero, and axle split.
            physics: Point-mass physical model settings.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If any provided
                parameter set is invalid.
        """
        self.physics = physics
        super().__init__(vehicle=vehicle, envelope_physics=physics.envelope)
        self.validate()

    def _validate_backend(self) -> None:
        """Validate point-mass-specific model configuration.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If point-mass
                physical parameters violate constraints.
        """
        self.physics.validate()


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


def calibrate_point_mass_friction_to_single_track(
    vehicle: VehicleParameters,
    tires: AxleTireParameters,
    single_track_physics: SingleTrackPhysics | None = None,
    single_track_numerics: SingleTrackNumerics | None = None,
    speed_samples: np.ndarray | None = None,
) -> PointMassCalibrationResult:
    """Calibrate point-mass friction coefficient to single_track lateral capability.

    The calibration matches the point-mass isotropic lateral limit
    ``mu * (g + F_down(v)/m)`` to the single_track model's quasi-steady lateral limit
    in a least-squares sense over provided speed samples.

    Args:
        vehicle: Vehicle parameterization shared by both models.
        tires: Tire parameters used by the single_track model.
        single_track_physics: Optional single_track-physics settings for calibration.
            Defaults to :class:`pylapsim.vehicle.SingleTrackPhysics`.
        single_track_numerics: Optional single_track numerical settings for calibration.
            Defaults to :class:`pylapsim.vehicle.SingleTrackNumerics`.
        speed_samples: Optional calibration speeds [m/s]. If omitted, a linear
            sweep from 10 to 90 m/s is used.

    Returns:
        Calibration result with identified friction coefficient and
        intermediate traces.

    Raises:
        pylapsim.utils.exceptions.ConfigurationError: If provided speed
            samples are empty or contain non-positive entries.
    """
    from pylapsim.vehicle.single_track_model import (
        SingleTrackModel,
        SingleTrackNumerics,
        SingleTrackPhysics,
    )

    single_track_model = SingleTrackModel(
        vehicle=vehicle,
        tires=tires,
        physics=single_track_physics or SingleTrackPhysics(),
        numerics=single_track_numerics or SingleTrackNumerics(),
    )
    single_track_model.validate()

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

    single_track_ay = np.asarray(
        single_track_model.lateral_accel_limit_batch(
            speed=speeds,
            banking=np.zeros_like(speeds, dtype=float),
        ),
        dtype=float,
    )
    if single_track_ay.shape != speeds.shape:
        msg = (
            "SingleTrackModel.lateral_accel_limit_batch returned an unexpected shape: "
            f"expected {speeds.shape}, got {single_track_ay.shape}"
        )
        raise ConfigurationError(msg)

    speed_sq = np.maximum(speeds, 0.0) ** 2
    downforce_scale = 0.5 * vehicle.air_density * vehicle.lift_coefficient * vehicle.frontal_area
    downforce = downforce_scale * speed_sq
    normal_accel = np.maximum(GRAVITY + downforce / vehicle.mass, SMALL_EPS)

    numerator = float(np.dot(normal_accel, single_track_ay))
    denominator = float(np.dot(normal_accel, normal_accel))
    mu_fit = max(numerator / max(denominator, SMALL_EPS), SMALL_EPS)
    mu_samples = single_track_ay / normal_accel

    return PointMassCalibrationResult(
        friction_coefficient=float(mu_fit),
        speed_samples=speeds,
        single_track_lateral_limit=single_track_ay,
        normal_accel_limit=normal_accel,
        mu_samples=mu_samples,
    )
